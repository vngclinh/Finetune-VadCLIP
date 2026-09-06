import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser(description="Temporal shift-consistency fine-tuning of VadCLIP on UCF-Crime")
parser.add_argument("--seed", default=234, type=int)
parser.add_argument(
    "--deterministic",
    default=False,
    type=str2bool,
    help="Pin cudnn to deterministic kernels and disable benchmarking, so re-running the "
         "same command on the same GPU gives the same weights. Mirrors the flag of the "
         "same name in baseline/src/ucf_train.py, which the control run has to be "
         "comparable with. Off by default, which is the upstream behaviour and what "
         "every run before this flag existed used.",
)

parser.add_argument("--embed-dim", default=512, type=int)
parser.add_argument("--visual-length", default=256, type=int)
parser.add_argument("--visual-width", default=512, type=int)
parser.add_argument("--visual-head", default=1, type=int)
parser.add_argument("--visual-layers", default=2, type=int)
parser.add_argument("--attn-window", default=8, type=int)
parser.add_argument("--prompt-prefix", default=10, type=int)
parser.add_argument("--prompt-postfix", default=10, type=int)
parser.add_argument("--classes-num", default=14, type=int)

parser.add_argument("--max-epoch", default=10, type=int)
parser.add_argument("--batch-size", default=64, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--scheduler-rate", default=0.1, type=float)
parser.add_argument("--scheduler-milestones", default=[4, 8], nargs="+", type=int)

# Shift-consistency specific.
parser.add_argument("--lambda-consistency", default=0.01, type=float)
parser.add_argument("--consistency-branch", default="c", choices=["c", "a", "both"])
parser.add_argument("--consistency-detach", default=False, type=str2bool, help="Freeze the full view as a fixed target.")
parser.add_argument("--consistency-warmup", default=1, type=int, help="Epochs of linear ramp-up from 0.")
parser.add_argument(
    "--skip-shifted-view",
    default=False,
    type=str2bool,
    help="Forward only the full view, halving the visual batch. Legal ONLY when the "
         "consistency term is identically zero for the whole run (the lambda-0 control), "
         "and the script refuses otherwise. CLIPVAD is batch independent -- "
         "tests/test_two_view_batching.py asserts exactly that -- so the full view's "
         "logits are the same either way, and a term multiplied by 0 contributes no "
         "gradient. Off by default so the control keeps taking the same code path as the "
         "runs it is a control for; turn it on when GPU hours are the binding constraint.",
)

# How the second view is cut. All four default to the original behaviour, so a command
# line that does not mention them reproduces the first round exactly.
parser.add_argument("--shift-offset", default=26, type=int, help="Temporal shift in grid steps (~10% of 256).")
parser.add_argument(
    "--shift-ratio",
    default=0.0,
    type=float,
    help="Size the shift as a fraction of each video's own valid length instead of a "
         "constant number of grid steps: offset = round(ratio * len). 0 keeps "
         "--shift-offset. A constant offset is a constant fraction of the GRID, not of "
         "the video -- 72%% of training files are shorter than the grid (median length "
         "138), so 26 steps means ~10%% for a long video and ~19%% for the median one. "
         "Sizing by length equalises the dose and leaves every video with an overlap.",
)
parser.add_argument(
    "--shift-direction",
    default="head",
    choices=["head", "tail", "both"],
    help="head: drop the first offset rows (content moves earlier, the original "
         "behaviour). tail: pad the front instead (content moves later, and nothing is "
         "lost at all when len + offset fits the grid). both: draw the side per item, so "
         "neither end of the video is systematically the one that gets cut.",
)
parser.add_argument(
    "--random-shift",
    default=False,
    type=str2bool,
    help="Sample the magnitude uniformly in [1, offset] per item instead of always using "
         "the maximum. A fixed offset only constrains invariance at that one distance.",
)
parser.add_argument(
    "--shift-ratio-warmup",
    default=0,
    type=int,
    help="Epochs of linear ramp-up on the shift magnitude, mirroring --consistency-warmup "
         "on lambda. 0 disables the curriculum. Turning it on forces persistent_workers "
         "off, because worker processes hold a pickled copy of the dataset.",
)

# Calibrating lambda by its share of the objective rather than by hand.
parser.add_argument(
    "--lambda-auto",
    default=0.0,
    type=float,
    help="Target ratio L_consistency_weighted / L_task. Runs --lambda-auto-steps steps "
         "with the term switched off, measures both sides, then solves "
         "lambda = ratio * L_task / L_consistency. 0 keeps --lambda-consistency. A ratio "
         "transfers across configurations in a way an absolute lambda does not.",
)
parser.add_argument("--lambda-auto-steps", default=50, type=int,
                    help="Free steps before --lambda-auto solves for lambda.")
parser.add_argument(
    "--lambda-auto-recalibrate",
    default=False,
    type=str2bool,
    help="Re-solve lambda at the end of every epoch instead of once per run. A single "
         "solve pins lambda to the task loss of an almost-untrained model; the task loss "
         "then falls several-fold while the consistency loss falls further, so the share "
         "the term actually holds drifts below the target. Round 2 asked for 0.01 / 0.03 "
         "/ 0.10 and ended at 1.9%% / 2.6%% / 4.0%%: a decade of separation collapsed to a "
         "factor of two. Off by default so round-2 runs stay reproducible.",
)
parser.add_argument(
    "--lambda-auto-max-growth",
    default=2.0,
    type=float,
    help="Bound on how far one recalibration may move lambda, as a multiplicative factor "
         "in both directions. Holding a fixed share against a consistency loss that is "
         "heading to zero demands an unbounded lambda, so the controller needs a leash. "
         "0 removes the bound. Ignored unless --lambda-auto-recalibrate is on.",
)

# Bookkeeping for a sweep.
parser.add_argument(
    "--select-metric",
    default="auc",
    choices=["auc", "none"],
    help="'auc' keeps the best-scoring weights out of every mid-training evaluation on "
         "the TEST set -- the upstream VadCLIP rule, kept as the default so old runs stay "
         "reproducible. 'none' keeps the final weights. Round 2 of the rescaling work "
         "measured the difference: the selection rule inflated the run-to-run spread by "
         "roughly an order of magnitude, so any sweep whose effect is smaller than a "
         "point of AUC needs 'none'.",
)
parser.add_argument("--run-tag", default="", help="Name for this run in the metrics CSV.")
parser.add_argument("--metrics-csv", default="", help="Append one row per evaluation here.")

parser.add_argument("--use-pretrained-model", default=False, type=str2bool)
parser.add_argument("--pretrained-model-path", default="../../model_ucf.pth")
parser.add_argument("--output-model-path", default="model/model_ucf_shift_consistency.pth")
parser.add_argument("--checkpoint-path", default="model/checkpoint_shift_consistency.pth")
parser.add_argument("--save-cur-path", default="model/model_cur_shift_consistency.pth")
parser.add_argument("--epoch-checkpoint-dir", default="model/epoch_checkpoints_shift_consistency")

parser.add_argument("--feature-root", default="../../UCFClipFeatures")
parser.add_argument("--train-list", default="../list/ucf_CLIP_rgb_relative.csv")
parser.add_argument("--test-list", default="../list/ucf_CLIP_rgbtest_relative.csv")
parser.add_argument("--gt-path", default="../list/gt_ucf.npy")
parser.add_argument("--gt-segment-path", default="../list/gt_segment_ucf.npy")
parser.add_argument("--gt-label-path", default="../list/gt_label_ucf.npy")

parser.add_argument("--eval-steps", default=1280, type=int, help="Matches the ucf_train.py evaluation cadence.")
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--pin-memory", default=True, type=str2bool)
parser.add_argument(
    "--debug-max-steps",
    default=0,
    type=int,
    help="Stop after N optimizer steps. Used by the step-5 equivalence check; 0 runs the full schedule.",
)
