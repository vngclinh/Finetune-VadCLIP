import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from model import CLIPVAD
from model_multi_prompt import CLIPVADMultiPrompt
from ucf_evaluate import UCFRelativeTestDataset
from utils.prompt_replacement import build_multi_prompt_groups, build_multi_prompt_weights
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

PROMPT_LABEL_MAP = {
    "Normal": "normal",
    "Abuse": "abuse",
    "Arrest": "arrest",
    "Arson": "arson",
    "Assault": "assault",
    "Burglary": "burglary",
    "Explosion": "explosion",
    "Fighting": "fighting",
    "RoadAccidents": "roadAccidents",
    "Robbery": "robbery",
    "Shooting": "shooting",
    "Shoplifting": "shoplifting",
    "Stealing": "stealing",
    "Vandalism": "vandalism",
}


def safe_metric(metric_fn, gt, scores):
    if len(np.unique(gt)) < 2:
        return math.nan
    return float(metric_fn(gt, scores) * 100)


def save_plot(path):
    plt.savefig(path, dpi=180)
    plt.close()
    print("Saved plot:", path)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_state_dict(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def build_lengths(length, visual_length):
    length_remaining = length
    lengths = torch.zeros(int(length / visual_length) + 1)
    for j in range(int(length / visual_length) + 1):
        if j == 0 and length_remaining < visual_length:
            lengths[j] = length_remaining
        elif j == 0 and length_remaining > visual_length:
            lengths[j] = visual_length
            length_remaining -= visual_length
        elif length_remaining > visual_length:
            lengths[j] = visual_length
            length_remaining -= visual_length
        else:
            lengths[j] = length_remaining
    return lengths.to(int)


def build_prompt_groups(args):
    return build_multi_prompt_groups(
        args.prompt_json,
        PROMPT_LABEL_MAP,
        "manual",
        prompt_mode=args.prompt_mode,
    )


def build_prompt_weights(args, device):
    return build_multi_prompt_weights(args.prompt_weights_json, PROMPT_LABEL_MAP, device)


def analyze_text_space(model, prompt_groups, output_dir):
    with torch.no_grad():
        grouped = model.encode_textprompt_groups(prompt_groups)
        prompt_features = F.normalize(grouped.float(), dim=-1).detach().cpu().numpy()

    class_names = list(PROMPT_LABEL_MAP.keys())
    prompts = []
    for class_index, (class_name, group) in enumerate(zip(class_names, prompt_groups)):
        for prompt_index, prompt in enumerate(group):
            prompts.append(
                {
                    "class_index": class_index,
                    "class_name": class_name,
                    "prompt_index": prompt_index,
                    "prompt_text": prompt,
                    "embedding": prompt_features[class_index, prompt_index],
                }
            )

    flat = np.stack([row["embedding"] for row in prompts], axis=0)
    class_features = prompt_features.mean(axis=1)
    class_features = class_features / np.maximum(np.linalg.norm(class_features, axis=1, keepdims=True), 1e-8)
    prompt_similarity = flat @ flat.T
    class_similarity = class_features @ class_features.T

    class_rows = []
    for i, source in enumerate(class_names):
        row = {"class_name": source}
        for j, target in enumerate(class_names):
            row[target] = float(class_similarity[i, j])
        class_rows.append(row)
    write_csv(output_dir / "class_text_similarity.csv", class_rows, ["class_name", *class_names])

    prompt_labels = [f"{row['class_name']}#{row['prompt_index']}" for row in prompts]
    prompt_rows = []
    for i, label in enumerate(prompt_labels):
        row = {"prompt": label}
        for j, target in enumerate(prompt_labels):
            row[target] = float(prompt_similarity[i, j])
        prompt_rows.append(row)
    write_csv(output_dir / "prompt_text_similarity.csv", prompt_rows, ["prompt", *prompt_labels])

    suspicious_rows = []
    for row_index, prompt in enumerate(prompts):
        own_class = prompt["class_index"]
        own_scores = prompt_similarity[row_index, own_class * len(prompt_groups[0]): (own_class + 1) * len(prompt_groups[0])]
        own_best = float(np.max(np.delete(own_scores, prompt["prompt_index"]))) if len(own_scores) > 1 else math.nan
        class_scores = flat[row_index] @ class_features.T
        other_order = np.argsort(class_scores)[::-1]
        for other_class in other_order:
            if other_class != own_class:
                best_other = int(other_class)
                break
        suspicious_rows.append(
            {
                "class_name": prompt["class_name"],
                "prompt_index": prompt["prompt_index"],
                "prompt_text": prompt["prompt_text"],
                "own_class_similarity": float(class_scores[own_class]),
                "best_other_class": class_names[best_other],
                "best_other_similarity": float(class_scores[best_other]),
                "other_minus_own": float(class_scores[best_other] - class_scores[own_class]),
                "nearest_same_class_prompt_similarity": own_best,
            }
        )
    suspicious_rows.sort(key=lambda item: item["other_minus_own"], reverse=True)
    write_csv(
        output_dir / "suspicious_prompt_matches.csv",
        suspicious_rows,
        [
            "class_name",
            "prompt_index",
            "prompt_text",
            "own_class_similarity",
            "best_other_class",
            "best_other_similarity",
            "other_minus_own",
            "nearest_same_class_prompt_similarity",
        ],
    )

    plot_heatmap(
        class_similarity,
        class_names,
        class_names,
        "Class Text Similarity",
        output_dir / "01_class_text_similarity_heatmap.png",
        figsize=(10, 8),
    )
    plot_heatmap(
        prompt_similarity,
        prompt_labels,
        prompt_labels,
        "Prompt Text Similarity",
        output_dir / "02_prompt_text_similarity_heatmap.png",
        figsize=(18, 16),
        tick_fontsize=4,
    )

    return prompts, class_similarity, prompt_similarity, suspicious_rows


def plot_heatmap(matrix, y_labels, x_labels, title, output_path, figsize=(10, 8), tick_fontsize=8):
    plt.figure(figsize=figsize)
    im = plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation=90, fontsize=tick_fontsize)
    plt.yticks(np.arange(len(y_labels)), y_labels, fontsize=tick_fontsize)
    plt.title(title)
    plt.tight_layout()
    save_plot(output_path)


def infer_baseline(model, dataloader, args, gt, device):
    prompt_text = get_prompt_text(LABEL_MAP)
    records = []
    gt_cursor = 0
    with torch.no_grad():
        for item in dataloader:
            visual = item[0].squeeze(0)
            label = item[1][0] if isinstance(item[1], (list, tuple)) else item[1]
            video_id = item[3][0] if isinstance(item[3], (list, tuple)) else item[3]
            length = int(item[2])
            len_cur = length
            if len_cur < args.visual_length:
                visual = visual.unsqueeze(0)
            visual = visual.to(device)
            lengths = build_lengths(length, args.visual_length)
            padding_mask = get_batch_mask(lengths, args.visual_length).to(device)
            _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])

            classifier_clip = torch.sigmoid(logits1[:len_cur].squeeze(-1)).detach().cpu().numpy()
            alignment_clip = (1 - logits2[:len_cur].softmax(dim=-1)[:, 0]).detach().cpu().numpy()
            classifier_frame = np.repeat(classifier_clip, 16)
            alignment_frame = np.repeat(alignment_clip, 16)
            gt_frame = gt[gt_cursor: gt_cursor + len(classifier_frame)]
            gt_cursor += len(classifier_frame)
            records.append(
                {
                    "video_id": video_id,
                    "label": label,
                    "gt": gt_frame,
                    "classifier": classifier_frame,
                    "alignment": alignment_frame,
                }
            )
    return records


def infer_manual(model, dataloader, args, gt, prompt_groups, prompt_weights, device):
    records = []
    prompt_score_chunks = []
    gt_chunks = []
    gt_cursor = 0
    with torch.no_grad():
        for item in dataloader:
            visual = item[0].squeeze(0)
            label = item[1][0] if isinstance(item[1], (list, tuple)) else item[1]
            video_id = item[3][0] if isinstance(item[3], (list, tuple)) else item[3]
            length = int(item[2])
            len_cur = length
            if len_cur < args.visual_length:
                visual = visual.unsqueeze(0)
            visual = visual.to(device)
            lengths = build_lengths(length, args.visual_length)
            padding_mask = get_batch_mask(lengths, args.visual_length).to(device)
            _, logits1, logits2, prompt_logits = model(
                visual,
                padding_mask,
                prompt_groups,
                lengths,
                return_prompt_logits=True,
                prompt_weights=prompt_weights,
            )
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            prompt_logits = prompt_logits.reshape(
                prompt_logits.shape[0] * prompt_logits.shape[1],
                prompt_logits.shape[2],
                prompt_logits.shape[3],
            )

            classifier_clip = torch.sigmoid(logits1[:len_cur].squeeze(-1)).detach().cpu().numpy()
            alignment_clip = (1 - logits2[:len_cur].softmax(dim=-1)[:, 0]).detach().cpu().numpy()
            prompt_clip = torch.sigmoid(prompt_logits[:len_cur]).detach().cpu().numpy()
            classifier_frame = np.repeat(classifier_clip, 16)
            alignment_frame = np.repeat(alignment_clip, 16)
            prompt_frame = np.repeat(prompt_clip, 16, axis=0)
            gt_frame = gt[gt_cursor: gt_cursor + len(classifier_frame)]
            gt_cursor += len(classifier_frame)
            prompt_score_chunks.append(prompt_frame)
            gt_chunks.append(gt_frame)
            records.append(
                {
                    "video_id": video_id,
                    "label": label,
                    "gt": gt_frame,
                    "classifier": classifier_frame,
                    "alignment": alignment_frame,
                    "prompt_scores": prompt_frame,
                }
            )
    return records, np.concatenate(prompt_score_chunks, axis=0), np.concatenate(gt_chunks)


def summarize_video(record):
    gt = record["gt"]
    result = {}
    for branch in ["classifier", "alignment"]:
        scores = record[branch]
        result[f"{branch}_mean"] = float(np.mean(scores))
        result[f"{branch}_max"] = float(np.max(scores))
        if np.any(gt == 1):
            result[f"{branch}_positive_mean"] = float(np.mean(scores[gt == 1]))
        else:
            result[f"{branch}_positive_mean"] = math.nan
        if np.any(gt == 0):
            result[f"{branch}_negative_mean"] = float(np.mean(scores[gt == 0]))
        else:
            result[f"{branch}_negative_mean"] = math.nan
        result[f"{branch}_auc"] = safe_metric(roc_auc_score, gt, scores)
        result[f"{branch}_ap"] = safe_metric(average_precision_score, gt, scores)
    return result


def analyze_score_drift(baseline_records, manual_records, output_dir):
    manual_lookup = {record["video_id"]: record for record in manual_records}
    rows = []
    for base in baseline_records:
        manual = manual_lookup[base["video_id"]]
        base_summary = summarize_video(base)
        manual_summary = summarize_video(manual)
        row = {
            "video_id": base["video_id"],
            "label": base["label"],
            "num_frames": len(base["gt"]),
            "positive_frames": int(np.sum(base["gt"])),
        }
        for key, value in base_summary.items():
            row[f"baseline_{key}"] = value
        for key, value in manual_summary.items():
            row[f"manual_{key}"] = value
            base_value = base_summary[key]
            if isinstance(base_value, float) and not (math.isnan(base_value) or math.isnan(value)):
                row[f"delta_{key}"] = value - base_value
            else:
                row[f"delta_{key}"] = math.nan
        rows.append(row)

    fieldnames = list(rows[0].keys())
    write_csv(output_dir / "per_video_score_drift.csv", rows, fieldnames)

    worst_rows = []
    for label in sorted({row["label"] for row in rows if row["label"] != "Normal"}):
        candidates = [row for row in rows if row["label"] == label]
        for sort_key in ["delta_classifier_auc", "delta_alignment_auc", "delta_classifier_positive_mean", "delta_alignment_positive_mean"]:
            valid = [row for row in candidates if not math.isnan(row.get(sort_key, math.nan))]
            for rank, row in enumerate(sorted(valid, key=lambda item: item[sort_key])[:5], 1):
                worst_rows.append({"label": label, "criterion": sort_key, "rank": rank, **row})
    write_csv(output_dir / "worst_video_drift_by_class.csv", worst_rows, ["label", "criterion", "rank", *fieldnames])

    plot_per_class_score_drift(rows, output_dir)
    return rows, worst_rows


def plot_per_class_score_drift(rows, output_dir):
    labels = sorted({row["label"] for row in rows if row["label"] != "Normal"})
    classifier_delta = []
    alignment_delta = []
    for label in labels:
        items = [row for row in rows if row["label"] == label]
        classifier_delta.append(np.nanmean([row["delta_classifier_positive_mean"] for row in items]))
        alignment_delta.append(np.nanmean([row["delta_alignment_positive_mean"] for row in items]))

    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(13, 6))
    plt.bar(x - width / 2, classifier_delta, width, label="classifier positive mean delta")
    plt.bar(x + width / 2, alignment_delta, width, label="alignment positive mean delta")
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Manual - baseline")
    plt.title("Per-Class Positive Score Drift")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    save_plot(output_dir / "03_per_class_score_drift.png")


def analyze_prompt_level(prompt_scores, gt, prompt_groups, suspicious_rows, output_dir):
    class_names = list(PROMPT_LABEL_MAP.keys())
    suspicious_lookup = {
        (row["class_name"], int(row["prompt_index"])): row for row in suspicious_rows
    }
    rows = []
    for class_index, class_name in enumerate(class_names):
        for prompt_index, prompt_text in enumerate(prompt_groups[class_index]):
            scores = prompt_scores[:, class_index, prompt_index]
            pos_scores = scores[gt == 1]
            neg_scores = scores[gt == 0]
            suspicious = suspicious_lookup[(class_name, prompt_index)]
            rows.append(
                {
                    "class_name": class_name,
                    "class_index": class_index,
                    "prompt_index": prompt_index,
                    "prompt_text": prompt_text,
                    "auc": safe_metric(roc_auc_score, gt, scores),
                    "ap": safe_metric(average_precision_score, gt, scores),
                    "positive_mean": float(np.mean(pos_scores)) if len(pos_scores) else math.nan,
                    "negative_mean": float(np.mean(neg_scores)) if len(neg_scores) else math.nan,
                    "margin": float(np.mean(pos_scores) - np.mean(neg_scores)) if len(pos_scores) and len(neg_scores) else math.nan,
                    "best_other_class": suspicious["best_other_class"],
                    "other_minus_own": suspicious["other_minus_own"],
                }
            )

    fieldnames = [
        "class_name",
        "class_index",
        "prompt_index",
        "prompt_text",
        "auc",
        "ap",
        "positive_mean",
        "negative_mean",
        "margin",
        "best_other_class",
        "other_minus_own",
    ]
    write_csv(output_dir / "prompt_level_metrics.csv", rows, fieldnames)

    suggestions = []
    for row in rows:
        reasons = []
        if row["margin"] < 0:
            reasons.append("negative_margin")
        if row["ap"] < 20:
            reasons.append("low_ap")
        if row["other_minus_own"] > -0.02:
            reasons.append("text_confusion")
        action = "drop_or_rewrite" if reasons else "keep"
        suggestions.append({**row, "suggested_action": action, "reasons": ";".join(reasons)})
    suggestions.sort(key=lambda item: (item["suggested_action"] != "drop_or_rewrite", item["margin"]))
    write_csv(output_dir / "prompt_keep_drop_suggestions.csv", suggestions, [*fieldnames, "suggested_action", "reasons"])

    plot_prompt_metric_heatmap(rows, "margin", output_dir / "04_prompt_margin_heatmap.png")
    plot_prompt_metric_heatmap(rows, "ap", output_dir / "05_prompt_ap_heatmap.png")
    return rows, suggestions


def analyze_prompt_level_by_video_class(manual_records, prompt_groups, output_dir):
    class_names = list(PROMPT_LABEL_MAP.keys())
    rows = []
    for eval_label in sorted({record["label"] for record in manual_records}):
        eval_records = [record for record in manual_records if record["label"] == eval_label]
        eval_gt = np.concatenate([record["gt"] for record in eval_records])
        eval_prompt_scores = np.concatenate([record["prompt_scores"] for record in eval_records], axis=0)
        for prompt_class_index, prompt_class_name in enumerate(class_names):
            for prompt_index, prompt_text in enumerate(prompt_groups[prompt_class_index]):
                scores = eval_prompt_scores[:, prompt_class_index, prompt_index]
                pos_scores = scores[eval_gt == 1]
                neg_scores = scores[eval_gt == 0]
                rows.append(
                    {
                        "eval_video_label": eval_label,
                        "prompt_class_name": prompt_class_name,
                        "prompt_class_index": prompt_class_index,
                        "prompt_index": prompt_index,
                        "prompt_text": prompt_text,
                        "auc": safe_metric(roc_auc_score, eval_gt, scores),
                        "ap": safe_metric(average_precision_score, eval_gt, scores),
                        "positive_mean": float(np.mean(pos_scores)) if len(pos_scores) else math.nan,
                        "negative_mean": float(np.mean(neg_scores)) if len(neg_scores) else math.nan,
                        "margin": float(np.mean(pos_scores) - np.mean(neg_scores)) if len(pos_scores) and len(neg_scores) else math.nan,
                    }
                )

    write_csv(
        output_dir / "prompt_level_metrics_by_video_class.csv",
        rows,
        [
            "eval_video_label",
            "prompt_class_name",
            "prompt_class_index",
            "prompt_index",
            "prompt_text",
            "auc",
            "ap",
            "positive_mean",
            "negative_mean",
            "margin",
        ],
    )
    return rows


def plot_prompt_metric_heatmap(rows, metric, output_path):
    class_names = list(PROMPT_LABEL_MAP.keys())
    prompt_count = max(row["prompt_index"] for row in rows) + 1
    matrix = np.full((len(class_names), prompt_count), np.nan)
    for row in rows:
        matrix[row["class_index"], row["prompt_index"]] = row[metric]
    plt.figure(figsize=(12, 7))
    im = plt.imshow(matrix, aspect="auto", cmap="coolwarm" if metric == "margin" else "viridis")
    plt.colorbar(im, label=metric)
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.xticks(np.arange(prompt_count), [str(i) for i in range(prompt_count)])
    plt.xlabel("Prompt index")
    plt.title(f"Prompt-Level {metric}")
    plt.tight_layout()
    save_plot(output_path)


def plot_worst_timelines(baseline_records, manual_records, worst_rows, output_dir, max_videos=12):
    timeline_dir = output_dir / "06_worst_class_timelines"
    timeline_dir.mkdir(parents=True, exist_ok=True)
    manual_lookup = {record["video_id"]: record for record in manual_records}
    selected = []
    seen = set()
    focus = {"Arrest", "Explosion", "Fighting", "Assault"}
    for row in worst_rows:
        if row["label"] not in focus:
            continue
        video_id = row["video_id"]
        if video_id not in seen:
            selected.append(row)
            seen.add(video_id)
        if len(selected) >= max_videos:
            break

    base_lookup = {record["video_id"]: record for record in baseline_records}
    for row in selected:
        base = base_lookup[row["video_id"]]
        manual = manual_lookup[row["video_id"]]
        x = np.arange(len(base["gt"])) / 16.0
        plt.figure(figsize=(14, 5))
        plt.plot(x, base["classifier"], label="baseline classifier", linewidth=1.2)
        plt.plot(x, manual["classifier"], label="manual classifier", linewidth=1.2)
        plt.plot(x, base["alignment"], label="baseline alignment", linewidth=1.0, linestyle="--")
        plt.plot(x, manual["alignment"], label="manual alignment", linewidth=1.0, linestyle="--")
        plt.fill_between(x, 0, base["gt"], color="tab:red", alpha=0.18, label="GT anomaly")
        plt.ylim(-0.03, 1.05)
        plt.xlabel("Time unit")
        plt.ylabel("Score")
        plt.title(f"{row['video_id']} ({row['label']}) - {row['criterion']}")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        safe_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in row["video_id"])
        save_plot(timeline_dir / f"{row['label']}_{safe_id}_{row['criterion']}.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze manual CLIP-friendly prompts for VadCLIP.")
    parser.add_argument("--feature-root", default="../../UCFClipFeatures")
    parser.add_argument("--test-list", default="../list/ucf_CLIP_rgbtest_relative.csv")
    parser.add_argument("--baseline-model-path", default="../../model_ucf.pth")
    parser.add_argument("--manual-model-path", default="model/model_ucf_multi_prompt_manual_caption_10x_scratch.pth")
    parser.add_argument("--prompt-json", default="../../code/ucf_manual_class_prompts_10x.json")
    parser.add_argument("--prompt-weights-json", default=None)
    parser.add_argument(
        "--prompt-mode",
        default="manual_caption",
        choices=["manual_caption", "class_plus_prototype", "prototype_only"],
    )
    parser.add_argument("--output-dir", default="../../code/ucf_manual_prompt_diagnostics")
    parser.add_argument("--gt-path", default="../list/gt_ucf.npy")
    parser.add_argument("--gt-segment-path", default="../list/gt_segment_ucf.npy")
    parser.add_argument("--gt-label-path", default="../list/gt_label_ucf.npy")
    parser.add_argument("--embed-dim", default=512, type=int)
    parser.add_argument("--visual-length", default=256, type=int)
    parser.add_argument("--visual-width", default=512, type=int)
    parser.add_argument("--visual-head", default=1, type=int)
    parser.add_argument("--visual-layers", default=2, type=int)
    parser.add_argument("--attn-window", default=8, type=int)
    parser.add_argument("--prompt-prefix", default=10, type=int)
    parser.add_argument("--prompt-postfix", default=10, type=int)
    parser.add_argument("--classes-num", default=14, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gt = np.load(args.gt_path)
    dataset = UCFRelativeTestDataset(args.visual_length, args.test_list, args.feature_root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    if args.batch_size != 1:
        raise ValueError("This analyzer expects --batch-size 1 so frame-level ground truth alignment is stable.")

    prompt_groups = build_prompt_groups(args)
    prompt_weights = build_prompt_weights(args, device)
    if len(prompt_groups) != args.classes_num:
        raise ValueError(f"Expected {args.classes_num} prompt groups, got {len(prompt_groups)}")
    print("Prompt groups:", len(prompt_groups), "prompts per class:", sorted({len(group) for group in prompt_groups}))

    manual_model = CLIPVADMultiPrompt(
        args.classes_num,
        args.embed_dim,
        args.visual_length,
        args.visual_width,
        args.visual_head,
        args.visual_layers,
        args.attn_window,
        args.prompt_prefix,
        args.prompt_postfix,
        device,
    )
    manual_model = load_state_dict(manual_model, args.manual_model_path, device)
    prompts, class_similarity, prompt_similarity, suspicious_rows = analyze_text_space(manual_model, prompt_groups, output_dir)

    print("Running baseline inference:", args.baseline_model_path)
    baseline_model = CLIPVAD(
        args.classes_num,
        args.embed_dim,
        args.visual_length,
        args.visual_width,
        args.visual_head,
        args.visual_layers,
        args.attn_window,
        args.prompt_prefix,
        args.prompt_postfix,
        device,
    )
    baseline_model = load_state_dict(baseline_model, args.baseline_model_path, device)
    baseline_records = infer_baseline(baseline_model, dataloader, args, gt, device)
    del baseline_model
    if device == "cuda":
        torch.cuda.empty_cache()

    print("Running manual prompt inference:", args.manual_model_path)
    manual_records, prompt_scores, prompt_gt = infer_manual(
        manual_model,
        dataloader,
        args,
        gt,
        prompt_groups,
        prompt_weights,
        device,
    )
    score_rows, worst_rows = analyze_score_drift(baseline_records, manual_records, output_dir)
    prompt_rows, suggestions = analyze_prompt_level(prompt_scores, prompt_gt, prompt_groups, suspicious_rows, output_dir)
    prompt_by_video_class_rows = analyze_prompt_level_by_video_class(manual_records, prompt_groups, output_dir)
    plot_worst_timelines(baseline_records, manual_records, worst_rows, output_dir)

    summary = {
        "num_test_videos": len(dataset),
        "num_prompt_groups": len(prompt_groups),
        "prompts_per_class": len(prompt_groups[0]),
        "num_suspicious_prompt_rows": len(suspicious_rows),
        "num_per_video_rows": len(score_rows),
        "num_prompt_metric_rows": len(prompt_rows),
        "num_prompt_by_video_class_rows": len(prompt_by_video_class_rows),
        "num_drop_or_rewrite_suggestions": sum(1 for row in suggestions if row["suggested_action"] == "drop_or_rewrite"),
    }
    with open(output_dir / "manual_prompt_diagnostics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved diagnostics to:", output_dir)


if __name__ == "__main__":
    main()
