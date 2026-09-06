"""Measure how much VadCLIP's frame scores move when the input is shifted in time.

This is the premise check for the shift-consistency direction: VadCLIP learns 256
independent absolute position embeddings and uses non-overlapping attention windows, so
the same event placed at a different offset can score differently. Nothing is trained
here; the script only probes a checkpoint.

For every test video the CLIP features go through ``tools.process_feat`` (the training
path, one 256-step grid per video), then each requested offset ``d`` drops the first
``d`` grid steps and zero-pads the tail. Scores of the shifted run are realigned onto the
original axis, so position ``j`` of the full view is compared against position ``j - d``
of the shifted view.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import utils.tools as tools
from model import CLIPVAD
from utils.dataset_augment import shift_view
from utils.tools import get_batch_mask, get_prompt_text


LABEL_MAP = {
    "Normal": "Normal",
    "Abuse": "Abuse",
    "Arrest": "Arrest",
    "Arson": "Arson",
    "Assault": "Assault",
    "Burglary": "Burglary",
    "Explosion": "Explosion",
    "Fighting": "Fighting",
    "RoadAccidents": "RoadAccidents",
    "Robbery": "Robbery",
    "Shooting": "Shooting",
    "Shoplifting": "Shoplifting",
    "Stealing": "Stealing",
    "Vandalism": "Vandalism",
}


def expand_grid_to_snippets(values, snippet_count, clip_dim):
    """Invert ``tools.uniform_extract``: map a clip_dim-long curve back onto raw snippets.

    ``uniform_extract`` averages original snippets ``r[i]:r[i+1]`` into grid step ``i``,
    so grid step ``i`` is broadcast back over exactly that range. Videos shorter than the
    grid were padded instead of resampled, so their first ``snippet_count`` steps are
    already one-to-one.
    """
    values = np.asarray(values)
    if snippet_count <= clip_dim:
        return values[:snippet_count].astype(np.float64)

    boundaries = np.linspace(0, snippet_count, clip_dim + 1, dtype=np.int32)
    expanded = np.zeros(snippet_count, dtype=np.float64)
    for i in range(clip_dim):
        start, end = boundaries[i], boundaries[i + 1]
        if start != end:
            expanded[start:end] = values[i]
        elif start < snippet_count:
            expanded[start] = values[i]
    return expanded


def load_model(args, device):
    model = CLIPVAD(
        args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
        args.visual_head, args.visual_layers, args.attn_window,
        args.prompt_prefix, args.prompt_postfix, device,
    )
    # weights_only=False so a full ucf_train.py-style checkpoint dict also loads; those carry
    # optimizer state that the PyTorch 2.6+ default refuses to unpickle.
    state = torch.load(args.model_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def score_all_offsets(model, feat_full, len_full, offsets, prompt_text, args, device):
    """Run every offset for one video in a single batched forward pass."""
    views = np.stack([shift_view(feat_full, offset) for offset in offsets], axis=0)
    visual = torch.tensor(views).to(device)

    lengths = torch.tensor(
        [max(1, min(len_full - offset, args.visual_length)) for offset in offsets], dtype=torch.long
    )
    padding_mask = get_batch_mask(lengths, args.visual_length).to(device)

    with torch.no_grad():
        _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths.to(device))
        classifier = torch.sigmoid(logits1.squeeze(-1)).float().cpu().numpy()
        alignment = (1 - logits2.softmax(dim=-1)[..., 0]).float().cpu().numpy()
    return classifier, alignment


def align_to_full_axis(curve, offset, len_full, clip_dim):
    """Return the shifted curve on the full view's axis, plus a validity mask."""
    aligned = np.zeros(clip_dim, dtype=np.float64)
    valid = np.zeros(clip_dim, dtype=bool)
    upper = min(len_full, clip_dim)
    if offset >= upper:
        return aligned, valid
    positions = np.arange(offset, upper)
    aligned[positions] = curve[positions - offset]
    valid[positions] = True
    return aligned, valid


def curve_agreement(reference, candidate, mask):
    """Pearson correlation and absolute deviation between two curves on the overlap."""
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan
    a = reference[mask]
    b = candidate[mask]
    deltas = np.abs(a - b)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        correlation = np.nan
    else:
        correlation = float(np.corrcoef(a, b)[0, 1])
    return correlation, float(deltas.mean()), float(deltas.max())


def plot_timelines(records, output_dir, offsets, count):
    ranked = sorted(
        (record for record in records if np.isfinite(record["worst_max_delta"])),
        key=lambda record: record["worst_max_delta"],
        reverse=True,
    )[:count]

    for rank, record in enumerate(ranked, start=1):
        figure, axis = plt.subplots(figsize=(11, 3.6))
        if record["snippet_gt"] is not None:
            axis.fill_between(
                np.arange(len(record["snippet_gt"])), 0, 1,
                where=record["snippet_gt"] > 0.5,
                color="#d62728", alpha=0.15, step="mid", label="ground truth",
            )
        for offset in offsets:
            curve = record["snippet_curves"][offset]
            axis.plot(np.arange(len(curve)), curve, linewidth=1.2, label=f"offset {offset}")
        axis.set_title(
            f"{record['video_id']} ({record['label']}) "
            f"max|delta|={record['worst_max_delta']:.3f}"
        )
        axis.set_xlabel("snippet index")
        axis.set_ylabel("anomaly score")
        axis.set_ylim(-0.02, 1.02)
        axis.legend(fontsize=8, ncol=len(offsets) + 1)
        figure.tight_layout()
        figure.savefig(Path(output_dir) / f"timeline_{rank:02d}_{record['video_id']}.png", dpi=150)
        plt.close(figure)


def main():
    args = parse_args()
    offsets = sorted(set(int(offset) for offset in args.offsets))
    if offsets[0] != 0:
        raise ValueError("--offsets must include 0; it is the unshifted reference.")
    if offsets[-1] >= args.visual_length:
        raise ValueError(f"--offsets must stay below --visual-length ({args.visual_length}).")
    max_offset = offsets[-1]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args, device)
    prompt_text = get_prompt_text(LABEL_MAP)

    with open(args.test_list, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ground_truth = None
    if Path(args.gt_path).exists():
        ground_truth = np.load(args.gt_path)
    else:
        print(f"WARNING: {args.gt_path} not found. Correlation stats only, no AUC.")

    feature_root = Path(args.feature_root)
    per_video_rows = []
    plot_records = []
    scores_by_offset = {offset: {"classifier": [], "alignment": [], "valid": []} for offset in offsets}
    gt_chunks = []
    gt_cursor = 0
    skipped = 0

    for row in rows:
        raw_feature = np.load(feature_root / row["path"])
        snippet_count = raw_feature.shape[0]
        feat_full, len_full = tools.process_feat(raw_feature, args.visual_length)
        len_full = int(len_full)

        gt_slice = None
        if ground_truth is not None:
            gt_slice = ground_truth[gt_cursor: gt_cursor + snippet_count * 16]
            gt_cursor += snippet_count * 16

        if len_full <= max_offset:
            skipped += 1
            continue

        classifier, alignment = score_all_offsets(
            model, feat_full, len_full, offsets, prompt_text, args, device
        )

        reference_classifier, reference_mask = align_to_full_axis(classifier[0], 0, len_full, args.visual_length)
        reference_alignment, _ = align_to_full_axis(alignment[0], 0, len_full, args.visual_length)

        snippet_curves = {}
        worst_max_delta = 0.0
        for index, offset in enumerate(offsets):
            aligned_classifier, valid = align_to_full_axis(classifier[index], offset, len_full, args.visual_length)
            aligned_alignment, _ = align_to_full_axis(alignment[index], offset, len_full, args.visual_length)
            overlap = valid & reference_mask

            correlation, mean_delta, max_delta = curve_agreement(
                reference_classifier, aligned_classifier, overlap
            )
            align_correlation, align_mean_delta, _ = curve_agreement(
                reference_alignment, aligned_alignment, overlap
            )
            if offset != 0 and np.isfinite(max_delta):
                worst_max_delta = max(worst_max_delta, max_delta)

            per_video_rows.append({
                "video_id": row.get("video_id", row["path"]),
                "label": row["label"],
                "offset": offset,
                "snippet_count": snippet_count,
                "len_full": len_full,
                "overlap_steps": int(overlap.sum()),
                "classifier_corr": correlation,
                "classifier_mean_abs_delta": mean_delta,
                "classifier_max_abs_delta": max_delta,
                "alignment_corr": align_correlation,
                "alignment_mean_abs_delta": align_mean_delta,
            })

            snippet_classifier = expand_grid_to_snippets(aligned_classifier, snippet_count, args.visual_length)
            snippet_curves[offset] = snippet_classifier
            if ground_truth is not None:
                snippet_valid = expand_grid_to_snippets(valid.astype(np.float64), snippet_count, args.visual_length) > 0.5
                snippet_alignment = expand_grid_to_snippets(aligned_alignment, snippet_count, args.visual_length)
                scores_by_offset[offset]["classifier"].append(np.repeat(snippet_classifier, 16))
                scores_by_offset[offset]["alignment"].append(np.repeat(snippet_alignment, 16))
                scores_by_offset[offset]["valid"].append(np.repeat(snippet_valid, 16))

        if ground_truth is not None:
            gt_chunks.append(gt_slice)

        snippet_gt = None
        if gt_slice is not None and len(gt_slice) == snippet_count * 16:
            snippet_gt = gt_slice.reshape(snippet_count, 16).max(axis=1)
        plot_records.append({
            "video_id": row.get("video_id", row["path"]),
            "label": row["label"],
            "worst_max_delta": worst_max_delta,
            "snippet_curves": snippet_curves,
            "snippet_gt": snippet_gt,
        })

    if not per_video_rows:
        raise RuntimeError("No test video was long enough for the requested offsets.")

    write_csv(output_dir / "shift_sensitivity.csv", per_video_rows)

    summary_rows = build_summary(
        per_video_rows, offsets, scores_by_offset, gt_chunks, ground_truth is not None
    )
    write_csv(output_dir / "shift_sensitivity_summary.csv", summary_rows)

    plot_timelines(plot_records, output_dir, offsets, args.timeline_count)

    print(f"\nVideos scored: {len(plot_records)} | skipped (shorter than max offset): {skipped}")
    print(f"Wrote {output_dir / 'shift_sensitivity.csv'}")
    print(f"Wrote {output_dir / 'shift_sensitivity_summary.csv'}")
    print_summary(summary_rows, ground_truth is not None)


def build_summary(per_video_rows, offsets, scores_by_offset, gt_chunks, has_gt):
    full_gt = np.concatenate(gt_chunks) if has_gt and gt_chunks else None
    common_mask = None
    if full_gt is not None:
        masks = [np.concatenate(scores_by_offset[offset]["valid"]) for offset in offsets]
        common_mask = np.logical_and.reduce(masks)

    summary_rows = []
    for offset in offsets:
        subset = [row for row in per_video_rows if row["offset"] == offset]
        row = {
            "offset": offset,
            "videos": len(subset),
            "mean_classifier_corr": float(np.nanmean([r["classifier_corr"] for r in subset])),
            "mean_classifier_abs_delta": float(np.nanmean([r["classifier_mean_abs_delta"] for r in subset])),
            "max_classifier_abs_delta": float(np.nanmax([r["classifier_max_abs_delta"] for r in subset])),
            "mean_alignment_corr": float(np.nanmean([r["alignment_corr"] for r in subset])),
            "mean_alignment_abs_delta": float(np.nanmean([r["alignment_mean_abs_delta"] for r in subset])),
        }
        if full_gt is not None:
            own_mask = np.concatenate(scores_by_offset[offset]["valid"])
            for branch in ("classifier", "alignment"):
                scores = np.concatenate(scores_by_offset[offset][branch])
                # Common region: identical frames for every offset, so AUCs are comparable.
                row[f"{branch}_auc_common"] = roc_auc_score(full_gt[common_mask], scores[common_mask]) * 100
                row[f"{branch}_auc_own"] = roc_auc_score(full_gt[own_mask], scores[own_mask]) * 100
        summary_rows.append(row)

    if full_gt is not None:
        for branch in ("classifier", "alignment"):
            values = [row[f"{branch}_auc_common"] for row in summary_rows]
            spread = max(values) - min(values)
            for row in summary_rows:
                row[f"{branch}_auc_spread_common"] = spread
    return summary_rows


def print_summary(summary_rows, has_gt):
    print("\nShift sensitivity summary (classifier branch)")
    header = f"{'offset':>7} {'corr':>8} {'mean|d|':>9} {'max|d|':>9}"
    if has_gt:
        header += f" {'AUC(common)':>12} {'AUC(own)':>10}"
    print(header)
    for row in summary_rows:
        line = (
            f"{row['offset']:>7} {row['mean_classifier_corr']:>8.4f} "
            f"{row['mean_classifier_abs_delta']:>9.4f} {row['max_classifier_abs_delta']:>9.4f}"
        )
        if has_gt:
            line += f" {row['classifier_auc_common']:>12.2f} {row['classifier_auc_own']:>10.2f}"
        print(line)
    if has_gt:
        spread = summary_rows[0]["classifier_auc_spread_common"]
        print(f"\nclassifier AUC spread across offsets (common region): {spread:.3f} points")
        print(f"alignment  AUC spread across offsets (common region): "
              f"{summary_rows[0]['alignment_auc_spread_common']:.3f} points")
        print(
            "\nDecision gate: corr < 0.95 or AUC spread > 0.5 -> premise holds, continue.\n"
            "               corr > 0.98 and AUC spread < 0.2 -> baseline already shift-invariant, report back."
        )


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Measure VadCLIP's sensitivity to temporal shifts.")
    parser.add_argument("--feature-root", default="../../UCFClipFeatures")
    parser.add_argument("--test-list", default="../list/ucf_CLIP_rgbtest_relative.csv")
    parser.add_argument("--model-path", default="../../model_ucf.pth")
    parser.add_argument("--gt-path", default="../list/gt_ucf.npy")
    parser.add_argument("--offsets", default=[0, 8, 16, 32], nargs="+", type=int)
    parser.add_argument("--output-dir", default="../../Result/ucf_shift_sensitivity")
    parser.add_argument("--timeline-count", default=5, type=int)

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


if __name__ == "__main__":
    main()
