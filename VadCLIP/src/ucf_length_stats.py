"""Report the snippet-length distribution of the UCF-Crime training list.

The shift-consistency design crops *after* ``process_feat``, so these numbers do not
change the design. They are needed for the data description in the report: videos with
``N <= 256`` keep one snippet per grid step, while longer videos are compressed by
``uniform_extract`` before the shift is applied.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


def summarise(name, lengths, clip_dim):
    if not lengths:
        print(f"\n{name}: no rows")
        return
    lengths = np.asarray(lengths)
    within_grid = int((lengths <= clip_dim).sum())
    print(f"\n{name}  (rows: {len(lengths)})")
    print(f"  min / max      : {lengths.min()} / {lengths.max()}")
    print(f"  mean / median  : {lengths.mean():.1f} / {np.median(lengths):.1f}")
    print(f"  N <= {clip_dim}       : {within_grid} ({100.0 * within_grid / len(lengths):.1f}%)")
    for offset in (8, 16, 26, 32, 51):
        too_short = int((lengths <= offset).sum())
        print(f"  N <= {offset:<3d} (no overlap at that offset): {too_short} "
              f"({100.0 * too_short / len(lengths):.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Snippet-length statistics for a UCF-Crime list.")
    parser.add_argument("--feature-root", default="../../UCFClipFeatures")
    parser.add_argument("--list-path", default="../list/ucf_CLIP_rgb_relative.csv")
    parser.add_argument("--visual-length", default=256, type=int)
    args = parser.parse_args()

    feature_root = Path(args.feature_root)
    with open(args.list_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    all_lengths = []
    normal_lengths = []
    abnormal_lengths = []
    missing = 0

    for row in rows:
        feature_path = feature_root / row["path"]
        if not feature_path.exists():
            missing += 1
            continue
        snippet_count = np.load(feature_path, mmap_mode="r").shape[0]
        all_lengths.append(snippet_count)
        if row["label"] == "Normal":
            normal_lengths.append(snippet_count)
        else:
            abnormal_lengths.append(snippet_count)

    print(f"List: {args.list_path}")
    print(f"Rows: {len(rows)} | readable: {len(all_lengths)} | missing files: {missing}")
    summarise("All", all_lengths, args.visual_length)
    summarise("Normal", normal_lengths, args.visual_length)
    summarise("Abnormal", abnormal_lengths, args.visual_length)


if __name__ == "__main__":
    main()
