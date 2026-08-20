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


parser = argparse.ArgumentParser(description="Class prototype-guided fine-tuning on UCF-Crime")
parser.add_argument("--seed", default=234, type=int)

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

parser.add_argument("--lambda-proto", default=0.05, type=float)
parser.add_argument("--lambda-proto-mil", default=1.0, type=float)
parser.add_argument("--prototype-temperature", default=0.07, type=float)
parser.add_argument("--prototype-mode", default="centroid", choices=["centroid"])
parser.add_argument("--prototype-json", default="../../code/ucf_class_description_prototypes.json")
parser.add_argument("--prototype-schema", default="long", choices=["long", "compact"])
parser.add_argument(
    "--compact-prototype-types",
    default=["keyword_phrase", "compact_action_phrase", "short_visual_sentence"],
    nargs="+",
    choices=["keyword_phrase", "compact_action_phrase", "short_visual_sentence"],
)

parser.add_argument("--use-pretrained-model", default=False, type=str2bool)
parser.add_argument("--pretrained-model-path", default="../../model_ucf.pth")
parser.add_argument("--output-model-path", default="model/model_ucf_class_prototype_scratch.pth")
parser.add_argument("--checkpoint-path", default="model/checkpoint_class_prototype_scratch.pth")

parser.add_argument("--feature-root", default="../../UCFClipFeatures")
parser.add_argument("--train-list", default="../list/ucf_CLIP_rgb_description.csv")
parser.add_argument("--test-list", default="../list/ucf_CLIP_rgbtest_relative.csv")
parser.add_argument("--gt-path", default="../list/gt_ucf.npy")
parser.add_argument("--gt-segment-path", default="../list/gt_segment_ucf.npy")
parser.add_argument("--gt-label-path", default="../list/gt_label_ucf.npy")

parser.add_argument("--eval-steps", default=0, type=int)
parser.add_argument("--save-cur-path", default="model/model_cur_class_prototype_scratch.pth")
parser.add_argument("--epoch-checkpoint-dir", default="model/epoch_checkpoints_class_prototype_scratch")
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--pin-memory", default=True, type=str2bool)
parser.add_argument("--use-amp", default=False, type=str2bool)
