"""Print the first-step losses of the original ucf_train.py data path (plan step 5).

ucf_train.py only logs at step 1280, so its first-step losses cannot be read without
editing it. This script reproduces that first step exactly instead: the same dataset
class (utils.dataset.UCFDataset), the same construction order, the same seed, and the
loss functions imported from ucf_train itself rather than copied.

Compare its output against the "[step 0]" line printed by

    python ucf_train_augment.py --lambda-consistency 0 --num-workers 0 --debug-max-steps 1

Both must use --num-workers 0 so the DataLoader draws from the RNG identically.
"""

import argparse
import csv
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import CLIPVAD
from ucf_train import CLAS2, CLASM, setup_seed
from utils.dataset import UCFDataset
from utils.tools import get_batch_label, get_prompt_text


def materialise_absolute_list(relative_list, feature_root, destination):
    """UCFDataset reads the path column directly, so it needs absolute paths."""
    with open(relative_list, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(destination, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"path": str(Path(feature_root) / row["path"]), "label": row["label"]})
    return destination


def parse_args():
    parser = argparse.ArgumentParser(description="Reference first-step losses from the ucf_train.py path.")
    parser.add_argument("--feature-root", default="../../UCFClipFeatures")
    parser.add_argument("--train-list", default="../list/ucf_CLIP_rgb_relative.csv")
    parser.add_argument("--test-list", default="../list/ucf_CLIP_rgbtest_relative.csv")
    parser.add_argument("--seed", default=234, type=int)
    parser.add_argument("--batch-size", default=64, type=int)

    parser.add_argument("--embed-dim", default=512, type=int)
    parser.add_argument("--visual-length", default=256, type=int)
    parser.add_argument("--visual-width", default=512, type=int)
    parser.add_argument("--visual-head", default=1, type=int)
    parser.add_argument("--visual-layers", default=2, type=int)
    parser.add_argument("--attn-window", default=8, type=int)
    parser.add_argument("--prompt-prefix", default=10, type=int)
    parser.add_argument("--prompt-postfix", default=10, type=int)
    parser.add_argument("--classes-num", default=14, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup_seed(args.seed)

    label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})

    temp_dir = Path(tempfile.mkdtemp())
    train_list = materialise_absolute_list(args.train_list, args.feature_root, temp_dir / "train_abs.csv")
    test_list = materialise_absolute_list(args.test_list, args.feature_root, temp_dir / "test_abs.csv")

    # Construction order copied from ucf_train.py so the RNG stream matches.
    normal_dataset = UCFDataset(args.visual_length, str(train_list), False, label_map, True)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    anomaly_dataset = UCFDataset(args.visual_length, str(train_list), False, label_map, False)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Built but unused: it consumes no RNG, and keeping it preserves the original order.
    test_dataset = UCFDataset(args.visual_length, str(test_list), True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLIPVAD(
        args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
        args.visual_head, args.visual_layers, args.attn_window,
        args.prompt_prefix, args.prompt_postfix, device,
    )
    model.to(device)
    model.train()

    print("normal videos:", len(normal_dataset),
          "| abnormal videos:", len(anomaly_dataset),
          "| test videos:", len(test_dataset),
          "| test batches:", len(test_loader))

    prompt_text = get_prompt_text(label_map)
    normal_iter = iter(normal_loader)
    anomaly_iter = iter(anomaly_loader)
    normal_features, normal_label, normal_lengths = next(normal_iter)
    anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)

    visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
    text_labels = list(normal_label) + list(anomaly_label)
    feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
    text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

    text_features, logits1, logits2 = model(visual_features, None, prompt_text, feat_lengths)

    loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
    loss2 = CLASM(logits2, text_labels, feat_lengths, device)
    loss3 = torch.zeros(1).to(device)
    text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
    for j in range(1, text_features.shape[0]):
        text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
        loss3 += torch.abs(text_feature_normal @ text_feature_abr)
    loss3 = loss3 / 13 * 1e-1

    print(
        f"[reference step 0] loss1={loss1.item():.8f} loss2={loss2.item():.8f} "
        f"loss3={loss3.item():.8f}"
    )


if __name__ == "__main__":
    main()
