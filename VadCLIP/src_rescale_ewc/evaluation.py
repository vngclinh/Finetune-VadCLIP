"""Scoring for stage-2 runs: overall metrics plus the per-class split the experiment needs.

The paper reports a two-sided table -- WER on the original benchmark (must not degrade)
next to recall on the target words (must improve). The counterpart here is overall AUC
next to per-class AUC/AP on the target classes, so ``evaluate`` returns both from a single
pass over the test set.

The forward pass, the chunk-length bookkeeping and the frame expansion (``np.repeat(...,
16)``) are copied from ``VadCLIP/src/ucf_test.py``; the per-class AUC follows
``ucf_analyze_checkpoints.per_class_metrics`` -- scores of one class' videos against those
same videos' frame labels -- so the numbers line up with the existing reports.
"""

import csv
import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def read_video_meta(test_list):
    """(video_id, label) in file order, which is also the order the test loader yields."""
    with open(test_list, "r", encoding="utf-8") as handle:
        return [(row.get("video_id", ""), row["label"]) for row in csv.DictReader(handle)]


def _chunk_lengths(length, maxlen):
    """Verbatim from ucf_test.py: split a video's clip count across its 256-frame chunks."""
    lengths = torch.zeros(int(length / maxlen) + 1)
    for j in range(int(length / maxlen) + 1):
        if j == 0 and length < maxlen:
            lengths[j] = length
        elif j == 0 and length > maxlen:
            lengths[j] = maxlen
            length -= maxlen
        elif length > maxlen:
            lengths[j] = maxlen
            length -= maxlen
        else:
            lengths[j] = length
    return lengths.to(int)


@torch.no_grad()
def score_test_set(model, testloader, maxlen, prompt_text, gt, video_meta, device):
    from utils.tools import get_batch_mask

    model.to(device)
    model.eval()

    records = []
    element_logits2_stack = []
    classifier_scores = []
    alignment_scores = []
    cursor = 0

    for index, item in enumerate(testloader):
        visual = item[0].squeeze(0)
        length = int(item[2])
        len_cur = length
        if len_cur < maxlen:
            visual = visual.unsqueeze(0)
        visual = visual.to(device)

        lengths = _chunk_lengths(length, maxlen)
        padding_mask = get_batch_mask(lengths, maxlen).to(device)
        _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
        logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
        logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])

        classifier_clip = torch.sigmoid(logits1[:len_cur].squeeze(-1)).detach().cpu().numpy()
        alignment_clip = (1 - logits2[:len_cur].softmax(dim=-1)[:, 0].squeeze(-1)).detach().cpu().numpy()
        classifier_frame = np.repeat(classifier_clip, 16)
        alignment_frame = np.repeat(alignment_clip, 16)
        gt_frame = gt[cursor: cursor + len(classifier_frame)]
        cursor += len(classifier_frame)

        classifier_scores.append(classifier_frame)
        alignment_scores.append(alignment_frame)
        element_logits2_stack.append(np.repeat(logits2[:len_cur].softmax(dim=-1).detach().cpu().numpy(), 16, 0))

        video_id, label = video_meta[index] if index < len(video_meta) else ("", "Unknown")
        records.append({
            "video_id": video_id,
            "label": label,
            "gt": gt_frame,
            "classifier": classifier_frame,
            "alignment": alignment_frame,
        })

    classifier_scores = np.concatenate(classifier_scores)
    alignment_scores = np.concatenate(alignment_scores)
    if len(classifier_scores) != len(gt):
        raise ValueError(
            f"Scored {len(classifier_scores)} frames but the ground truth has {len(gt)}. "
            "The test list and gt_ucf.npy must describe the same videos in the same order."
        )
    return records, classifier_scores, alignment_scores, element_logits2_stack


def _safe_auc_ap(gt, scores):
    if len(np.unique(gt)) < 2:
        return math.nan, math.nan
    return roc_auc_score(gt, scores) * 100, average_precision_score(gt, scores) * 100


def per_class_metrics(records, branch):
    """One row per label; a class' videos are scored against their own frame labels."""
    rows = {}
    for label in sorted({record["label"] for record in records}):
        subset = [record for record in records if record["label"] == label]
        auc, ap = _safe_auc_ap(
            np.concatenate([record["gt"] for record in subset]),
            np.concatenate([record[branch] for record in subset]),
        )
        rows[label] = {"auc": auc, "ap": ap}
    return rows


def _group_mean(per_class, labels, key):
    values = [per_class[label][key] for label in labels
              if label in per_class and not math.isnan(per_class[label][key])]
    return float(np.mean(values)) if values else math.nan


def evaluate(model, testloader, maxlen, prompt_text, gt, gtsegments, gtlabels,
             video_meta, target_classes, device, with_map=True):
    """Everything one stage-2 run needs to be judged, from a single pass."""
    from utils.ucf_detectionMAP import getDetectionMAP as dmAP

    records, classifier_scores, alignment_scores, element_logits2_stack = score_test_set(
        model, testloader, maxlen, prompt_text, gt, video_meta, device
    )

    metrics = {}
    metrics["classifier_auc"], metrics["classifier_ap"] = _safe_auc_ap(gt, classifier_scores)
    metrics["alignment_auc"], metrics["alignment_ap"] = _safe_auc_ap(gt, alignment_scores)

    if with_map:
        dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
        for threshold, value in zip(iou, dmap):
            metrics[f"mAP@{threshold:.1f}"] = float(value)
        metrics["avg_mAP"] = float(np.mean(dmap))

    per_class = {branch: per_class_metrics(records, branch) for branch in ("classifier", "alignment")}

    # The two halves of the paper's trade-off table: the emphasised set, and everything else.
    anomaly_labels = [label for label in per_class["classifier"] if label != "Normal"]
    target = [label for label in anomaly_labels if label in set(target_classes)]
    rest = [label for label in anomaly_labels if label not in set(target_classes)]
    for branch, prefix in (("classifier", "c"), ("alignment", "a")):
        metrics[f"target_auc_{prefix}"] = _group_mean(per_class[branch], target, "auc")
        metrics[f"target_ap_{prefix}"] = _group_mean(per_class[branch], target, "ap")
        metrics[f"rest_auc_{prefix}"] = _group_mean(per_class[branch], rest, "auc")
        metrics[f"rest_ap_{prefix}"] = _group_mean(per_class[branch], rest, "ap")

    # False-positive rate on the 150 Normal test videos, at the usual 0.5 threshold. This
    # is the "precision" column of the paper's tables: the cost side of emphasising classes.
    normal_scores = [record["classifier"] for record in records if record["label"] == "Normal"]
    metrics["normal_fpr@0.5"] = (
        float((np.concatenate(normal_scores) > 0.5).mean() * 100) if normal_scores else math.nan
    )

    return metrics, per_class, records


def format_summary(metrics, target_classes):
    targets = ", ".join(target_classes)
    return (
        f"  overall  | AUC_C {metrics['classifier_auc']:6.2f} | AP_C {metrics['classifier_ap']:6.2f}"
        f" | AUC_A {metrics['alignment_auc']:6.2f} | mAP {metrics.get('avg_mAP', float('nan')):6.2f}\n"
        f"  target   | AUC_C {metrics['target_auc_c']:6.2f} | AP_C {metrics['target_ap_c']:6.2f}"
        f" | AUC_A {metrics['target_auc_a']:6.2f}   ({targets})\n"
        f"  non-targ | AUC_C {metrics['rest_auc_c']:6.2f} | AP_C {metrics['rest_ap_c']:6.2f}"
        f" | AUC_A {metrics['rest_auc_a']:6.2f}\n"
        f"  normals  | FPR@0.5 {metrics['normal_fpr@0.5']:6.2f}%"
    )


def append_metrics_csv(path, row):
    """Append one row, reusing an existing header so several runs can share one file.

    If the file already has a header (a different run, possibly an older version of this
    script), that header wins: unknown keys are dropped and missing ones are left blank,
    rather than silently writing rows whose columns no longer line up.
    """
    header = None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            header = next(csv.reader(handle), None)
    except FileNotFoundError:
        pass

    with open(path, "a", newline="", encoding="utf-8") as handle:
        if header:
            writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
            writer.writerow({key: row.get(key, "") for key in header})
        else:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
