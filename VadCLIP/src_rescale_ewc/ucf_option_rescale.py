"""Command-line options for stage-2 loss-rescaling + EWC fine-tuning of VadCLIP.

Model/data defaults mirror ``VadCLIP/src/ucf_option_augment.py`` so a run differs from
the baseline only in the parts this experiment is about. The stage-2 training defaults
(lr, clipping, epochs) follow arXiv:2302.09723 section 4.5, where fine-tuning uses a much
smaller learning rate and tighter gradient clipping than the run that produced the source
model.
"""

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


# The four lowest-AUC classes that still have enough test videos (>= 20) for a per-class
# AUC to mean anything. Abuse scores just as badly but has only 2 test videos, so it is
# deliberately left out -- optimising against a 2-video metric is how you fool yourself.
DEFAULT_TARGET_CLASSES = ["Explosion", "RoadAccidents", "Shooting", "Shoplifting"]

UCF_LABEL_MAP = {
    "Normal": "normal", "Abuse": "abuse", "Arrest": "arrest", "Arson": "arson",
    "Assault": "assault", "Burglary": "burglary", "Explosion": "explosion",
    "Fighting": "fighting", "RoadAccidents": "roadAccidents", "Robbery": "robbery",
    "Shooting": "shooting", "Shoplifting": "shoplifting", "Stealing": "stealing",
    "Vandalism": "vandalism",
}

parser = argparse.ArgumentParser(
    description="Stage-2 VadCLIP fine-tuning with loss rescaling and weight consolidation "
                "(arXiv:2302.09723 applied to UCF-Crime anomaly classes)"
)

parser.add_argument("--seed", default=234, type=int)

# --- Architecture (must match the stage-1 checkpoint) --------------------------------
parser.add_argument("--embed-dim", default=512, type=int)
parser.add_argument("--visual-length", default=256, type=int)
parser.add_argument("--visual-width", default=512, type=int)
parser.add_argument("--visual-head", default=1, type=int)
parser.add_argument("--visual-layers", default=2, type=int)
parser.add_argument("--attn-window", default=8, type=int)
parser.add_argument("--prompt-prefix", default=10, type=int)
parser.add_argument("--prompt-postfix", default=10, type=int)
parser.add_argument("--classes-num", default=14, type=int)

# --- Stage-2 optimisation -------------------------------------------------------------
# Paper section 4.5: lr drops from 4e-3 to 4e-6 and clipping from 5 to 2 when fine-tuning,
# "since a tiny learning rate can efficiently ensure stable model learning and retain the
# previously learned knowledge". VadCLIP stage 1 runs at 2e-5, so 2e-6 keeps the same
# spirit over a much shorter schedule.
parser.add_argument("--max-epoch", default=3, type=int)
parser.add_argument("--batch-size", default=64, type=int,
                    help="Do not lower this while rescaling: paper section 5.4 shows a small batch "
                         "can end up entirely made of target samples, which blows the gradient up.")
parser.add_argument("--lr", default=2e-6, type=float)
parser.add_argument("--scheduler-rate", default=0.1, type=float)
parser.add_argument("--scheduler-milestones", default=[2], nargs="+", type=int)
parser.add_argument("--grad-clip", default=1.0, type=float, help="0 disables clipping.")

# --- Loss rescaling, Eq. (1) and Eq. (10)-(11) ---------------------------------------
parser.add_argument("--target-classes", default=DEFAULT_TARGET_CLASSES, nargs="+",
                    help="Raw CSV labels to emphasise. The counterpart of the paper's OOV set O.")
parser.add_argument("--mu", default=1.0, type=float,
                    help="Rescaling factor. 1.0 disables rescaling. Useful range here is 2-10; the "
                         "paper's 100-10000 suits unbounded CTC losses, not these O(1) losses.")
parser.add_argument("--rescale-mode", default="class", choices=["off", "video", "class"],
                    help="'video' = Eq. (1), scale the whole per-video loss of target videos. "
                         "'class' = Eq. (10)-(11), scale only the gradient reaching the target "
                         "classes' A-branch logits, leaving the loss value and the C branch alone.")
parser.add_argument("--rescale-normalize", default="none", choices=["none", "mean"],
                    help="'none' averages the rescaled losses over the batch, as Eq. (1) does, so "
                         "the gradient scale grows with mu. 'mean' divides by sum(w) instead.")

# --- Weight consolidation, Eq. (12) and Eq. (13) -------------------------------------
parser.add_argument("--regularizer", default="ewc", choices=["none", "l2", "ewc"])
parser.add_argument("--lambda-reg", default=0.0, type=float,
                    help="Weight of the consolidation term. Calibrate it, do not copy the paper's "
                         "1e7: see --fisher-normalize and the L_reg/L_task line the trainer logs.")
parser.add_argument("--fisher-path", default="model/fisher_ucf.pt")
parser.add_argument("--fisher-normalize", default="mean", choices=["none", "mean", "max"])
parser.add_argument("--lambda-auto", default=0.0, type=float,
                    help="Target L_reg/L_task ratio. When > 0, lambda is solved for after "
                         "--lambda-auto-steps steps of free drift instead of being guessed. "
                         "0.10-0.20 is a sensible ask; 0 keeps the fixed --lambda-reg.")
parser.add_argument("--lambda-auto-steps", default=50, type=int,
                    help="Steps of unconstrained drift used to measure the penalty's scale.")

# --- Fisher estimation (ucf_fisher.py) ------------------------------------------------
parser.add_argument("--fisher-max-samples", default=1000, type=int,
                    help="Number of (normal, anomaly) pairs used for Eq. (14). 0 = whole train list.")
parser.add_argument("--fisher-log-every", default=100, type=int)

# --- Target-class oversampling, the counterpart of the paper's mixing ratio ----------
parser.add_argument("--target-oversample", default=1.0, type=float,
                    help="Sampling weight for target-class rows inside the anomaly loader. "
                         "1.0 = off. The normal/anomaly halves stay 50/50 regardless.")

# --- Checkpoints ---------------------------------------------------------------------
parser.add_argument("--pretrained-model-path", default="model/model_ucf.pth",
                    help="theta': the converged stage-1 model. Required unless "
                         "--use-pretrained-model false; stage-2 fine-tuning makes no sense "
                         "without a source model to anchor to.")
parser.add_argument(
    "--use-pretrained-model",
    default=True,
    type=str2bool,
    help="true (the default) loads --pretrained-model-path, which is what stage 2 means: "
         "fine-tune a converged model. false leaves the VadCLIP-specific layers at their "
         "random initialisation and trains them from scratch on top of the frozen CLIP "
         "encoder, matching --use-pretrained-model false in src/ucf_train_augment.py. "
         "From scratch there is no theta' to consolidate towards, so --regularizer must "
         "be 'none', and the stage-2 schedule no longer applies: use the baseline's "
         "--max-epoch 10 --lr 2e-5 --scheduler-milestones 4 8 --grad-clip 0 instead of "
         "one epoch at 2e-6. A number produced this way is NOT comparable with the "
         "stage-2 runs, which all started from a model that was already at 88.02.",
)
parser.add_argument("--output-model-path", default="model/model_ucf_rescale_ewc.pth")
parser.add_argument("--checkpoint-path", default="model/checkpoint_rescale_ewc.pth")
parser.add_argument("--save-cur-path", default="model/model_cur_rescale_ewc.pth")
parser.add_argument("--epoch-checkpoint-dir", default="model/epoch_checkpoints_rescale_ewc")

# --- Data ----------------------------------------------------------------------------
parser.add_argument("--feature-root", default="../../UCFClipFeatures")
parser.add_argument("--train-list", default="../list/ucf_CLIP_rgb_relative.csv")
parser.add_argument("--test-list", default="../list/ucf_CLIP_rgbtest_relative.csv")
parser.add_argument("--gt-path", default="../list/gt_ucf.npy")
parser.add_argument("--gt-segment-path", default="../list/gt_segment_ucf.npy")
parser.add_argument("--gt-label-path", default="../list/gt_label_ucf.npy")

# --- Evaluation / logging -------------------------------------------------------------
parser.add_argument("--eval-steps", default=1280, type=int, help="0 evaluates only at epoch ends.")
parser.add_argument("--run-tag", default="", help="Name for this run in the metrics CSV, so a sweep "
                                                  "can be compared without matching on flags.")
parser.add_argument("--select-metric", default="none",
                    choices=["none", "classifier_auc", "alignment_auc", "target_auc_c"],
                    help="Which metric picks the checkpoint to keep. 'none' (the default) keeps the "
                         "FINAL weights and never looks at the test set to choose. That is the honest "
                         "setting for a controlled sweep: best-of-N selection over many mid-training "
                         "evaluations on the test set is what put a 0.58 AUC noise floor under the "
                         "previous experiment. Stage 2 is only 3 epochs at a tiny lr, so there is no "
                         "overfitting story that justifies picking a peak.")
parser.add_argument("--metrics-csv", default="", help="Append per-evaluation metrics here.")
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--pin-memory", default=True, type=str2bool)
parser.add_argument("--debug-max-steps", default=0, type=int,
                    help="Stop after N optimizer steps. Used by the smoke test; 0 runs the schedule.")

# --- ucf_eval_perclass.py only --------------------------------------------------------
parser.add_argument("--eval-model-paths", default=[], nargs="+",
                    help="Checkpoints to score. Each may be 'name=path' or just a path.")
parser.add_argument("--eval-output", default="model/rescale_ewc_perclass.csv")
