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


parser = argparse.ArgumentParser(description="Description-guided VadCLIP fine-tuning on UCF-Crime")
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

parser.add_argument("--max-epoch", default=5, type=int)
parser.add_argument("--batch-size", default=64, type=int)
parser.add_argument("--lr", default=1e-5, type=float)
parser.add_argument("--scheduler-rate", default=0.1, type=float)
parser.add_argument("--scheduler-milestones", default=[3, 4], nargs="+", type=int)

parser.add_argument("--lambda-desc", default=0.003, type=float)
parser.add_argument("--lambda-contrastive", default=0.003, type=float)
parser.add_argument("--lambda-distill", default=0.05, type=float)
parser.add_argument("--desc-loss-type", default="contrastive", choices=["cosine", "contrastive"])
parser.add_argument("--desc-pooling", default="topk", choices=["mean", "topk", "weighted"])
parser.add_argument("--top-k-ratio", default=0.15, type=float)
parser.add_argument("--use-desc-projection", default=True, type=str2bool)
parser.add_argument("--distill-baseline", default=True, type=str2bool)
parser.add_argument("--contrastive-temperature", default=0.07, type=float)
parser.add_argument("--contrastive-samples", default="all", choices=["all", "abnormal"])
parser.add_argument(
    "--trainable-scope",
    default="all",
    choices=["all", "projection_only", "projection_mlp_classifier"],
)
parser.add_argument("--use-pretrained-model", default=True, type=str2bool)
parser.add_argument("--pretrained-model-path", default="../../model_ucf.pth")
parser.add_argument("--output-model-path", default="model/model_ucf_description.pth")
parser.add_argument("--checkpoint-path", default="model/checkpoint_description.pth")

parser.add_argument("--feature-root", default="../../UCFClipFeatures")
parser.add_argument("--description-json", default="../../code/ucf_gpt_video_descriptions.json")
parser.add_argument("--train-list", default="../list/ucf_CLIP_rgb_description.csv")
parser.add_argument("--test-list", default="../list/ucf_CLIP_rgbtest_relative.csv")
parser.add_argument("--gt-path", default="../list/gt_ucf.npy")
parser.add_argument("--gt-segment-path", default="../list/gt_segment_ucf.npy")
parser.add_argument("--gt-label-path", default="../list/gt_label_ucf.npy")

parser.add_argument("--eval-steps", default=0, type=int)
parser.add_argument("--save-cur-path", default="model/model_cur_description.pth")
parser.add_argument("--epoch-checkpoint-dir", default="model/epoch_checkpoints")
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--pin-memory", default=True, type=str2bool)
parser.add_argument("--use-amp", default=True, type=str2bool)
parser.add_argument("--cache-description-embeddings", default=True, type=str2bool)
