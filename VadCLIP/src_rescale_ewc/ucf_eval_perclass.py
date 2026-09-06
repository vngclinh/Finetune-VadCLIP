"""Score one or more checkpoints and print the trade-off table the experiment is judged on.

The layout mirrors the paper's Tables 3 and 4: the columns that must not degrade sit next
to the columns that should improve, so a run is read as a trade-off rather than as a
single number.

    python ucf_eval_perclass.py --feature-root /path/to/UCFClipFeatures \
        --eval-model-paths source=model/model_ucf.pth \
                           ctrl=model/model_ctrl.pth \
                           class_mu3=model/model_ucf_rescale_ewc.pth \
        --eval-output model/rescale_ewc_perclass.csv
"""

import csv

import numpy as np
import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401  (registers VadCLIP/src on sys.path)
import ucf_option_rescale
from evaluation import evaluate, read_video_meta


def parse_model_specs(specs):
    parsed = []
    for spec in specs:
        name, _, path = spec.partition("=")
        parsed.append((name, path) if path else (name, name))
    return parsed


def print_table(rows, target_classes):
    targets = ", ".join(target_classes)
    print("\n" + "=" * 108)
    print("MUST NOT DEGRADE (the 'WER' half)              |  SHOULD IMPROVE (the 'recall' half)")
    print("=" * 108)
    header = (f"{'run':<16}{'AUC_C':>8}{'AP_C':>8}{'AUC_A':>8}{'mAP':>8}{'restAUC':>9}"
              f"{'FPR%':>7}  |{'tgtAUC_C':>10}{'tgtAP_C':>9}{'tgtAUC_A':>10}{'tgtAP_A':>9}")
    print(header)
    print("-" * 108)
    for name, metrics in rows:
        print(f"{name:<16}{metrics['classifier_auc']:>8.2f}{metrics['classifier_ap']:>8.2f}"
              f"{metrics['alignment_auc']:>8.2f}{metrics.get('avg_mAP', float('nan')):>8.2f}"
              f"{metrics['rest_auc_c']:>9.2f}{metrics['normal_fpr@0.5']:>7.2f}  |"
              f"{metrics['target_auc_c']:>10.2f}{metrics['target_ap_c']:>9.2f}"
              f"{metrics['target_auc_a']:>10.2f}{metrics['target_ap_a']:>9.2f}")
    print("-" * 108)
    print(f"target classes: {targets}")

    if len(rows) > 1:
        base_name, base = rows[0]
        print(f"\nDelta against '{base_name}':")
        print(f"{'run':<16}{'dAUC_C':>9}{'drestAUC':>10}{'dmAP':>8}  |{'dTgtAUC_C':>11}{'dTgtAP_C':>10}{'dTgtAUC_A':>11}")
        for name, metrics in rows[1:]:
            print(f"{name:<16}{metrics['classifier_auc'] - base['classifier_auc']:>+9.2f}"
                  f"{metrics['rest_auc_c'] - base['rest_auc_c']:>+10.2f}"
                  f"{metrics.get('avg_mAP', float('nan')) - base.get('avg_mAP', float('nan')):>+8.2f}  |"
                  f"{metrics['target_auc_c'] - base['target_auc_c']:>+11.2f}"
                  f"{metrics['target_ap_c'] - base['target_ap_c']:>+10.2f}"
                  f"{metrics['target_auc_a'] - base['target_auc_a']:>+11.2f}")
        print("\nReminder: two mathematically identical runs of this pipeline previously differed by")
        print("0.58 AUC and 1.55 mAP. Overall deltas below that are noise, not results.")


def main():
    from model import CLIPVAD
    from utils.tools import get_prompt_text

    from dataset_rescale import UCFRescaleDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option_rescale.parser.parse_args()
    if not args.eval_model_paths:
        raise SystemExit("Pass at least one checkpoint via --eval-model-paths name=path")

    label_map = ucf_option_rescale.UCF_LABEL_MAP
    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)
    video_meta = read_video_meta(args.test_list)

    test_dataset = UCFRescaleDataset(args.visual_length, args.test_list, True, args.feature_root)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLIPVAD(
        args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
        args.visual_head, args.visual_layers, args.attn_window,
        args.prompt_prefix, args.prompt_postfix, device,
    )

    summary_rows = []
    per_class_rows = []
    for name, path in parse_model_specs(args.eval_model_paths):
        state = torch.load(path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        print(f"\nScoring {name} ({path}) ...", flush=True)

        metrics, per_class, _ = evaluate(
            model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels,
            video_meta, args.target_classes, device,
        )
        summary_rows.append((name, metrics))
        for branch, table in per_class.items():
            for label, values in table.items():
                per_class_rows.append({
                    "run": name, "branch": branch, "label": label,
                    "is_target": label in set(args.target_classes),
                    "auc": values["auc"], "ap": values["ap"],
                })

    print_table(summary_rows, args.target_classes)

    with open(args.eval_output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "branch", "label", "is_target", "auc", "ap"])
        writer.writeheader()
        writer.writerows(per_class_rows)
    print("\nSaved per-class metrics:", args.eval_output)

    summary_path = args.eval_output.replace(".csv", "_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = ["run"] + list(summary_rows[0][1].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, metrics in summary_rows:
            writer.writerow({"run": name, **metrics})
    print("Saved summary metrics:", summary_path)


if __name__ == "__main__":
    main()
