"""Measure per-class difficulty on the TRAIN set, then write the adaptive weight vector.

Step 1 of the two-step adaptive-rescaling pipeline:

    ucf_train_difficulty.py  --pretrained-model-path theta'.pth  -> adaptive_weights.json
    ucf_train_rescale.py     --rescale-mode adaptive_class --class-weight-file that.json

Splitting it in two is what makes the method auditable. The weight vector is a file you
can print, diff between seeds and drop into the paper, and every training run reads the
same frozen copy, so no run can quietly disagree with the table you reported.

Nothing here touches the test set. That is the whole point: the four "target" classes were
originally chosen because they scored worst *on test*, which puts the test set inside the
design of the method. Frequency and train-set difficulty put it back outside.

Weakly-supervised training gives no frame labels on train, so there is no train-set AUC to
measure. What there is, is exactly what the trainer optimises: the per-video MIL losses.
This script reads them through ``losses.clas2_per_video`` / ``losses.clasm_per_video`` --
the same functions the trainer reduces -- rather than reimplementing the top-k rule.

    mean_CLASM_loss           A-branch cross-entropy. The loss the rescaling acts on.
    mean_classifier_loss      C-branch BCE.
    mean_alignment_margin     true class logit minus its strongest rival, both top-k
                              pooled. Small margin = the class is nearly confused.
    mean_classifier_score_gap |target - MIL score|, so 0 is a perfect video either way.

One caveat the script prints rather than hides: the source checkpoint was trained on this
very train set, so a class with few videos received few gradient updates and keeps a high
train loss. Difficulty may therefore be frequency wearing a different hat. The Spearman
correlation between the two is printed with the table, and it is the number that decides
whether the beta ablation is measuring two signals or one.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

import _bootstrap  # noqa: F401  (registers VadCLIP/src on sys.path)
import adaptive_weights
import ucf_option_rescale
from losses import clas2_per_video, clasm_per_video

# How each measurement turns into "difficulty": higher must mean harder. A margin is the
# one that runs the other way -- a large margin is an easy class -- so it is negated.
DIFFICULTY_SOURCES = {
    "clasm_loss": ("mean_CLASM_loss", 1.0),
    "clas2_loss": ("mean_classifier_loss", 1.0),
    "alignment_margin": ("mean_alignment_margin", -1.0),
    "classifier_gap": ("mean_classifier_score_gap", 1.0),
}


def count_train_videos(train_list):
    """Unique ``video_id`` per class. The CSV has one row per crop, ten crops per video."""
    counts = defaultdict(set)
    with open(train_list, "r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            counts[row["label"]].add(row.get("video_id") or row["path"])
    return {label: len(video_ids) for label, video_ids in counts.items()}


@torch.no_grad()
def measure_class_statistics(model, loader, prompt_text, label_map, device, max_batches=0):
    """Per-class sums of the four measurements, plus the row count they average over.

    The model is put in eval mode and no gradients are taken: this is a measurement of a
    finished checkpoint, not a training step. Batch composition does not enter any of the
    numbers -- every one of them is computed per video, and CLIPVAD is batch independent
    (``tests/test_two_view_batching.py``) -- so the batch size here is free to differ from
    the trainer's.
    """
    from utils.tools import get_batch_label

    model.to(device)
    model.eval()

    totals = defaultdict(lambda: defaultdict(float))
    for index, (features, raw_labels, lengths) in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        visual = features.to(device)
        lengths = lengths.to(device)
        raw_labels = list(raw_labels)
        labels = get_batch_label(raw_labels, prompt_text, label_map).to(device)

        # padding_mask=None mirrors the training forward pass, not the test one: this is
        # meant to measure difficulty under the conditions the class was learned in.
        _, logits1, logits2 = model(visual, None, prompt_text, lengths)

        classifier_loss, mil_score, targets = clas2_per_video(logits1, labels, lengths, device)
        alignment_loss, instance_logits = clasm_per_video(logits2, labels, lengths, device)

        true_index = labels.argmax(dim=1)
        true_logit = instance_logits.gather(1, true_index.unsqueeze(1)).squeeze(1)
        rivals = instance_logits.clone()
        rivals.scatter_(1, true_index.unsqueeze(1), float("-inf"))
        margin = true_logit - rivals.max(dim=1).values
        score_gap = (targets - mil_score).abs()

        for position, label in enumerate(raw_labels):
            bucket = totals[label]
            bucket["rows"] += 1.0
            bucket["mean_CLASM_loss"] += float(alignment_loss[position])
            bucket["mean_classifier_loss"] += float(classifier_loss[position])
            bucket["mean_alignment_margin"] += float(margin[position])
            bucket["mean_classifier_score_gap"] += float(score_gap[position])

    return {label: dict(bucket) for label, bucket in totals.items()}


def average_statistics(totals):
    """Turn the accumulated sums into means, keeping the row count alongside."""
    averages = {}
    for label, bucket in totals.items():
        rows = bucket["rows"]
        averages[label] = {"rows": int(rows), **{
            key: value / rows for key, value in bucket.items() if key != "rows"}}
    return averages


def difficulty_from_statistics(averages, source):
    """Project the per-class statistics onto one 'higher is harder' number."""
    if source not in DIFFICULTY_SOURCES:
        raise ValueError(f"Unknown --difficulty-source {source!r}. "
                         f"Choose from {sorted(DIFFICULTY_SOURCES)}.")
    column, sign = DIFFICULTY_SOURCES[source]
    return {label: sign * stats[column] for label, stats in averages.items()}


def write_statistics_csv(path, averages, breakdown):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    columns = ["class_name", "train_video_count", "train_rows", "mean_CLASM_loss",
               "mean_classifier_loss", "mean_alignment_margin", "mean_classifier_score_gap",
               "difficulty_raw", "difficulty_score", "frequency_raw", "frequency_score",
               "mix", "adaptive_weight"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for label in sorted(breakdown, key=lambda name: -breakdown[name]["adaptive_weight"]):
            row = breakdown[label]
            stats = averages.get(label, {})
            writer.writerow({
                "class_name": label,
                "train_video_count": row["train_video_count"],
                "train_rows": stats.get("rows", 0),
                "mean_CLASM_loss": _round(stats.get("mean_CLASM_loss")),
                "mean_classifier_loss": _round(stats.get("mean_classifier_loss")),
                "mean_alignment_margin": _round(stats.get("mean_alignment_margin")),
                "mean_classifier_score_gap": _round(stats.get("mean_classifier_score_gap")),
                "difficulty_raw": _round(row.get("difficulty_raw")),
                "difficulty_score": _round(row.get("difficulty_score")),
                "frequency_raw": _round(row.get("frequency_raw")),
                "frequency_score": _round(row.get("frequency_score")),
                "mix": _round(row.get("mix")),
                "adaptive_weight": _round(row.get("adaptive_weight")),
            })


def _round(value, digits=6):
    return "" if value is None else round(float(value), digits)


def read_statistics_file(path):
    """Reuse an earlier run's measurements: ``(averages, counts, source_model)``.

    Only the arithmetic after the measurement depends on alpha/beta/w-max, so the whole
    beta ablation comes out of one pass over the train set instead of three.
    """
    with open(path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    if "statistics" not in document or "breakdown" not in document:
        raise SystemExit(f"{path} has no stored statistics. Only files written by this "
                         f"script (schema vadclip-adaptive-class-weights/1) can be reused.")
    counts = {label: row["train_video_count"] for label, row in document["breakdown"].items()}
    return document["statistics"], counts, document.get("source_model", str(path))


def measure_from_model(args, device):
    """Load theta', run it over the whole train list, return per-class averages."""
    from model import CLIPVAD
    from utils.tools import get_prompt_text

    from dataset_rescale import UCFRescaleDataset

    checkpoint = Path(args.pretrained_model_path)
    if not checkpoint.exists():
        raise SystemExit(
            f"No source checkpoint at {checkpoint}. Difficulty is a property of a trained "
            f"model, so this script has nothing to measure without one. Point "
            f"--pretrained-model-path at the stage-1 model the stage-2 runs will start from."
        )

    label_map = ucf_option_rescale.UCF_LABEL_MAP
    prompt_text = get_prompt_text(label_map)

    # Both halves of the train list, in one pass. UCFRescaleDataset splits Normal from the
    # rest for the trainer's two loaders; difficulty needs every row exactly once.
    normal_rows = UCFRescaleDataset(args.visual_length, args.train_list, False,
                                    args.feature_root, normal=True)
    anomaly_rows = UCFRescaleDataset(args.visual_length, args.train_list, False,
                                     args.feature_root, normal=False)
    loader = DataLoader(ConcatDataset([normal_rows, anomaly_rows]), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        pin_memory=bool(args.pin_memory and device == "cuda"))

    model = CLIPVAD(
        args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
        args.visual_head, args.visual_layers, args.attn_window,
        args.prompt_prefix, args.prompt_postfix, device,
    )
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    print("Source model theta':", checkpoint)
    print(f"Train rows: {len(normal_rows) + len(anomaly_rows)} "
          f"({len(normal_rows)} normal + {len(anomaly_rows)} anomaly)", flush=True)

    totals = measure_class_statistics(model, loader, prompt_text, label_map, device,
                                      args.difficulty_max_batches)
    return average_statistics(totals), count_train_videos(args.train_list), str(checkpoint)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option_rescale.parser.parse_args()

    if args.from_statistics:
        averages, counts, source_model = read_statistics_file(args.from_statistics)
        print(f"Reusing measurements from {args.from_statistics} (source model {source_model}); "
              f"only the weight arithmetic is recomputed.")
    else:
        averages, counts, source_model = measure_from_model(args, device)

    difficulty = difficulty_from_statistics(averages, args.difficulty_source)

    weights, breakdown = adaptive_weights.compute_adaptive_weights(
        counts, difficulty,
        alpha=args.adaptive_alpha, w_max=args.adaptive_w_max, beta=args.adaptive_beta,
        frequency_power=args.adaptive_frequency_power,
    )

    ranked = {name: counts[name] for name in breakdown
              if name not in adaptive_weights.PINNED_CLASSES}
    correlation = adaptive_weights.spearman(
        ranked, {name: breakdown[name]["difficulty_raw"] for name in ranked})

    print()
    print(adaptive_weights.format_weight_table(breakdown, correlation))
    print()

    document = adaptive_weights.build_weight_document(
        weights, breakdown,
        hyperparameters={
            "alpha": args.adaptive_alpha, "w_max": args.adaptive_w_max,
            "beta": args.adaptive_beta, "frequency_power": args.adaptive_frequency_power,
            "difficulty_source": args.difficulty_source,
        },
        extra={
            "source_model": source_model,
            "train_list": args.train_list,
            "statistics": {label: {key: round(value, 6) for key, value in stats.items()
                                   if key != "rows"} | {"rows": stats["rows"]}
                           for label, stats in sorted(averages.items())},
            "spearman_count_vs_difficulty": None if correlation is None else round(correlation, 6),
        },
    )
    Path(args.difficulty_output).parent.mkdir(parents=True, exist_ok=True)
    adaptive_weights.save_weight_document(args.difficulty_output, document)
    print("Wrote weights:", args.difficulty_output)

    if args.difficulty_csv:
        write_statistics_csv(args.difficulty_csv, averages, breakdown)
        print("Wrote statistics:", args.difficulty_csv)


if __name__ == "__main__":
    main()
