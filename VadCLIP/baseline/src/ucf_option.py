import argparse


def str2bool(value):
    """Upstream declares --use-checkpoint as type=bool, which makes argparse treat any
    non-empty string as True ('--use-checkpoint False' would enable it). Defaults are
    unchanged; only the parsing of an explicitly passed value is fixed."""
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('yes', 'true', 't', '1', 'y'):
        return True
    if value in ('no', 'false', 'f', '0', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='VadCLIP')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=2, type=int)
parser.add_argument('--attn-window', default=8, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=14, type=int)

parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--model-path', default='model/model_ucf.pth')
parser.add_argument('--use-checkpoint', default=False, type=str2bool)   # FIXED: upstream type=bool made every non-empty string True
parser.add_argument('--checkpoint-path', default='model/checkpoint.pth')
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--train-list', default='../../list/ucf_CLIP_rgb.csv')
parser.add_argument('--test-list', default='../../list/ucf_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='../../list/gt_ucf.npy')
parser.add_argument('--gt-segment-path', default='../../list/gt_segment_ucf.npy')
parser.add_argument('--gt-label-path', default='../../list/gt_label_ucf.npy')

# type=float added: upstream leaves these untyped, so '--lr 1e-5' would reach AdamW as the
# string '1e-5' and crash. The defaults below are the upstream values, unchanged.
parser.add_argument('--lr', default=2e-5, type=float)                   # FIXED: untyped upstream, so '--lr 1e-5' reached AdamW as a string
parser.add_argument('--scheduler-rate', default=0.1, type=float)
parser.add_argument('--scheduler-milestones', default=[4, 8], nargs='+', type=int)

# --- Additions required to run outside the author's machine. None of them touch the
# --- training math; see baseline/README.md for the full list.

# Prefix joined onto relative entries of the feature list. Empty (the default) keeps the
# upstream behaviour of reading the absolute paths stored in the CSV.
parser.add_argument('--feature-root', default='')                       # ADDED: prefix for relative feature lists; '' = upstream behaviour

# Archive a copy of the weights after every epoch. Upstream only keeps model/model_cur.pth,
# which is overwritten each epoch, so a dropped Colab session loses the whole run.
# Empty (the default) disables it and reproduces the upstream file layout exactly.
parser.add_argument('--epoch-checkpoint-dir', default='')               # ADDED: per-epoch weight archive; '' = off, upstream layout
parser.add_argument('--save-cur-path', default='model/model_cur.pth')   # ADDED: upstream hardcoded this string

# ADDED: DataLoader knobs. Defaults (0 / False) are the upstream DataLoader defaults, so
# leaving them alone reproduces the original data order exactly.
#
# WARNING: --num-workers changes WHICH videos land in WHICH batch, and therefore changes the
# resulting model. It is not a speed-only flag. Two runs that differ only in this number are
# two different draws, not the same run made faster. Record the value you used.
parser.add_argument('--num-workers', default=0, type=int)
parser.add_argument('--pin-memory', default=False, type=str2bool)

# ADDED: make a run reproducible on the same GPU. Upstream has the cudnn line commented out,
# so re-running the identical command gives a slightly different model every time.
# Costs a little speed. Default False = upstream behaviour.
parser.add_argument('--deterministic', default=False, type=str2bool)
