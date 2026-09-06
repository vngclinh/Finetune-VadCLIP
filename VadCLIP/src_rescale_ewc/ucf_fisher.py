"""Estimate the diagonal Fisher information of a converged VadCLIP model (Eq. 14).

Run this once, on the stage-1 checkpoint, before any stage-2 fine-tuning:

    python ucf_fisher.py --pretrained-model-path model/model_ucf.pth \
                         --feature-root /path/to/UCFClipFeatures \
                         --fisher-path model/fisher_ucf.pt

The output file carries both F and theta', which is everything Eq. (13) needs.

Two details that decide whether F is right or garbage:

* The loss differentiated here is the **unmodified stage-1 objective** L1+L2+L3. F has to
  describe the *source* task, so no rescaling is applied at this point.
* One sample is one **(normal video, anomaly video) pair**, and the squaring happens per
  sample. CLAS2 is a binary cross-entropy that needs both polarities to reflect the shape
  of the real training step, and Eq. (14) squares before summing -- averaging gradients
  first would let opposite signs cancel and bias F downwards.
"""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401  (registers VadCLIP/src on sys.path)
import ucf_option_rescale
from fisher import (
    accumulate_squared_gradients,
    clone_anchor,
    finalize_fisher,
    fisher_statistics,
    module_importance,
    normalize_fisher,
    save_fisher,
    trainable_named_parameters,
    zero_fisher,
)
from losses import CLAS2, CLASM, text_separation_loss


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def estimate(model, normal_loader, anomaly_loader, prompt_text, label_map, args, device):
    from utils.tools import get_batch_label

    model.to(device)
    model.train()  # matches the stage-1 conditions; VadCLIP has no dropout or batch norm

    fisher = zero_fisher(model)
    normal_iter = cycle(normal_loader)
    anomaly_iter = cycle(anomaly_loader)

    total = args.fisher_max_samples
    if total <= 0:
        total = min(len(normal_loader), len(anomaly_loader))
    print(f"Estimating the Fisher diagonal over {total} (normal, anomaly) pairs.")

    for step in range(total):
        normal_features, normal_label, normal_length = next(normal_iter)
        anomaly_features, anomaly_label, anomaly_length = next(anomaly_iter)

        visual = torch.cat([normal_features, anomaly_features], dim=0).to(device)
        lengths = torch.cat([normal_length, anomaly_length], dim=0).to(device)
        labels = get_batch_label(list(normal_label) + list(anomaly_label), prompt_text, label_map).to(device)

        text_features, logits1, logits2 = model(visual, None, prompt_text, lengths)
        loss = (
            CLAS2(logits1, labels, lengths, device)
            + CLASM(logits2, labels, lengths, device)
            + text_separation_loss(text_features, device).squeeze()
        )

        model.zero_grad(set_to_none=True)
        accumulate_squared_gradients(model, loss, fisher)

        if args.fisher_log_every and (step + 1) % args.fisher_log_every == 0:
            print(f"  {step + 1}/{total} pairs | last loss {loss.item():.6f}", flush=True)

    return finalize_fisher(fisher, total), total


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option_rescale.parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    from model import CLIPVAD
    from utils.tools import get_prompt_text

    from dataset_rescale import UCFRescaleDataset

    label_map = ucf_option_rescale.UCF_LABEL_MAP
    prompt_text = get_prompt_text(label_map)

    normal_dataset = UCFRescaleDataset(args.visual_length, args.train_list, False, args.feature_root, normal=True)
    anomaly_dataset = UCFRescaleDataset(args.visual_length, args.train_list, False, args.feature_root, normal=False)
    # batch_size 1: Eq. (14) squares each sample's gradient before averaging.
    normal_loader = DataLoader(normal_dataset, batch_size=1, shuffle=True, drop_last=True)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=1, shuffle=True, drop_last=True)

    model = CLIPVAD(
        args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
        args.visual_head, args.visual_layers, args.attn_window,
        args.prompt_prefix, args.prompt_postfix, device,
    )
    state = torch.load(args.pretrained_model_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    print("Loaded source model theta':", args.pretrained_model_path)

    trainable = trainable_named_parameters(model)
    print(f"Trainable tensors: {len(trainable)} | trainable parameters: "
          f"{sum(p.numel() for _, p in trainable):,} "
          f"(the CLIP backbone is frozen and is excluded)")

    fisher, used = estimate(model, normal_loader, anomaly_loader, prompt_text, label_map, args, device)
    anchor = clone_anchor(model, device="cpu")

    # Save before reporting. The estimation loop above is the only expensive part of this
    # script, and nothing printed below is worth losing it over.
    fisher_cpu = {name: value.cpu() for name, value in fisher.items()}
    meta = {
        "source_model": args.pretrained_model_path,
        "pairs": used,
        "train_list": args.train_list,
        "seed": args.seed,
        "note": "Raw (un-normalised) diagonal Fisher; normalisation happens at load time.",
    }
    save_fisher(args.fisher_path, fisher_cpu, anchor, meta)
    print("\nSaved Fisher + anchor:", args.fisher_path)

    print("\nFisher statistics (raw):")
    for key, value in fisher_statistics(fisher).items():
        print(f"  {key:>16}: {value}")

    print("\nMean Fisher per module (raw, top 15) -- what the source task leans on:")
    for module, mean_value, count in module_importance(fisher):
        print(f"  {module:>28}: {mean_value:.6e}  ({count:,} params)")

    normalized = normalize_fisher(fisher, "mean")
    print("\nAfter --fisher-normalize mean, the statistics lambda will actually see:")
    for key, value in fisher_statistics(normalized).items():
        print(f"  {key:>16}: {value}")


if __name__ == "__main__":
    main()
