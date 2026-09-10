"""Stage-2 VadCLIP fine-tuning: loss rescaling + weight consolidation (arXiv:2302.09723).

This script never trains from scratch. It starts from a converged stage-1 VadCLIP model
theta', which serves two roles at once, exactly as in the paper: the initialisation of
fine-tuning, and the anchor the consolidation term pulls back towards.

    L = L1' + L2' + L3 + (lambda/2) * sum_i F_i (theta_i - theta'_i)^2

where L1'/L2' are the C- and A-branch MIL losses with the target classes emphasised, by
one of two mechanisms:

    --rescale-mode video   Eq. (1)        multiply the whole loss of a target-class video
    --rescale-mode class   Eq. (10)-(11)  multiply only the gradient reaching the target
                                          classes' A-branch logits; the loss value and the
                                          entire C branch are left alone

``model.py`` and ``utils/tools.py`` are untouched.
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, WeightedRandomSampler

import _bootstrap  # noqa: F401  (registers VadCLIP/src on sys.path)
import ucf_option_rescale
from evaluation import append_metrics_csv, evaluate, format_summary, read_video_meta
from fisher import load_fisher
from adaptive_weights import load_class_weights
from losses import (
    CLAS2,
    CLASM,
    build_class_scale,
    build_class_scale_from_weights,
    build_video_weights,
    consolidation_penalty,
    consolidation_scale,
    mean_absolute_drift,
    text_separation_loss,
)


def load_source_weights(model, path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)


def build_anomaly_sampler(dataset, target_classes, oversample):
    """The counterpart of the paper's source:target mixing ratio (Tables 1 and 2).

    The normal and anomaly halves of a batch stay 50/50 -- the two loaders are zipped, not
    concatenated -- so this only shifts the composition *inside* the anomaly half.
    """
    if oversample == 1.0:
        return None
    labels = dataset.labels
    weights = [oversample if label in target_classes else 1.0 for label in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    from utils.tools import get_batch_label, get_prompt_text

    model.to(device)
    for path in (args.checkpoint_path, args.output_model_path, args.save_cur_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.epoch_checkpoint_dir).mkdir(parents=True, exist_ok=True)

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)
    video_meta = read_video_meta(args.test_list)

    if args.use_pretrained_model:
        load_source_weights(model, args.pretrained_model_path, device)
        print("Loaded source model theta':", args.pretrained_model_path)
    else:
        # Same starting point as src/ucf_train_augment.py with --use-pretrained-model
        # false: the CLIP encoder keeps its released weights and stays frozen, every
        # VadCLIP-specific layer starts from its random initialisation.
        print("Training VadCLIP-specific layers from scratch (CLIP weights only).")
        if args.regularizer != "none" and (args.lambda_reg != 0 or args.lambda_auto > 0):
            raise SystemExit(
                "--regularizer " + args.regularizer + " needs a source model to pull back "
                "towards, but --use-pretrained-model is false, so there is no theta'. "
                "Pass --regularizer none, or drop --use-pretrained-model false."
            )

    prompt_text = get_prompt_text(label_map)
    target_classes = set(args.target_classes)

    # --- Consolidation term ----------------------------------------------------------
    fisher, anchor = None, {}
    if args.regularizer != "none" and (args.lambda_reg != 0 or args.lambda_auto > 0):
        fisher, anchor, fisher_meta = load_fisher(args.fisher_path, device, args.fisher_normalize)
        if args.regularizer == "l2":
            fisher = None  # Eq. (12) weighs every parameter equally
        print(f"Consolidation: {args.regularizer} | lambda {args.lambda_reg} "
              f"| fisher_normalize {args.fisher_normalize} | fisher meta {fisher_meta}")
    else:
        print("Consolidation: off (regularizer=none, or both lambda-reg and lambda-auto are 0).")

    # --- Rescaling -------------------------------------------------------------------
    class_scale = None
    if args.rescale_mode == "class" and args.mu != 1.0:
        class_scale = build_class_scale(prompt_text, label_map, target_classes, args.mu, device)
    elif args.rescale_mode == "adaptive_class":
        class_weights, weight_meta = load_class_weights(args.class_weight_file)
        class_scale = build_class_scale_from_weights(prompt_text, label_map, class_weights, device)
        args.adaptive_beta_used = weight_meta.get("beta", "")
        print(f"Adaptive weights from {args.class_weight_file} | {weight_meta}")
        for raw_label, weight in sorted(class_weights.items(), key=lambda item: -item[1]):
            print(f"    {raw_label:<16} {weight:.3f}")
    print(f"Rescaling: mode {args.rescale_mode} | mu {args.mu} | normalize {args.rescale_normalize}")
    # In adaptive_class mode this set no longer steers the rescaling -- every class has its
    # own weight -- but it still decides which classes the report aggregates as 'target'.
    print(f"Target classes (reporting{'' if args.rescale_mode != 'adaptive_class' else ' only'}): "
          f"{sorted(target_classes)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)

    best_auc = 0.0
    global_step = 0
    lambda_reg = args.lambda_reg
    if args.lambda_auto > 0 and anchor:
        lambda_reg = 0.0
        print(f"Lambda auto-calibration on: after {args.lambda_auto_steps} free steps, lambda "
              f"will be solved for so that L_reg/L_task = {args.lambda_auto}.")

    for epoch in range(args.max_epoch):
        model.train()
        totals = {"l1": 0.0, "l2": 0.0, "l3": 0.0, "reg": 0.0}
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        num_steps = min(len(normal_loader), len(anomaly_loader))

        for i in range(num_steps):
            normal_features, normal_label, normal_length = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_length = next(anomaly_iter)

            visual = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            lengths = torch.cat([normal_length, anomaly_length], dim=0).to(device)
            raw_labels = list(normal_label) + list(anomaly_label)
            labels = get_batch_label(raw_labels, prompt_text, label_map).to(device)

            # Eq. (1) weights, only in 'video' mode. In 'class' mode the emphasis lives
            # entirely in the backward pass, so the forward loss is the untouched one.
            video_weights = None
            if args.rescale_mode == "video" and args.mu != 1.0:
                video_weights = build_video_weights(raw_labels, target_classes, args.mu, device)

            text_features, logits1, logits2 = model(visual, None, prompt_text, lengths)

            loss1 = CLAS2(logits1, labels, lengths, device, video_weights, args.rescale_normalize)
            loss2 = CLASM(logits2, labels, lengths, device, video_weights, args.rescale_normalize, class_scale)
            loss3 = text_separation_loss(text_features, device).squeeze()
            loss_reg = consolidation_penalty(model, anchor, fisher, lambda_reg)

            loss = loss1 + loss2 + loss3 + loss_reg

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            totals["l1"] += loss1.item()
            totals["l2"] += loss2.item()
            totals["l3"] += loss3.item()
            totals["reg"] += float(loss_reg.detach())

            if global_step == 0:
                print(f"[step 0] l1={loss1.item():.8f} l2={loss2.item():.8f} "
                      f"l3={loss3.item():.8f} reg={float(loss_reg.detach()):.8f}  "
                      f"(reg is 0 here by construction: theta == theta')")
            global_step += 1

            # Solve for lambda rather than hand-searching it across five orders of
            # magnitude, which is the paper's main practical weakness. The penalty is
            # (lambda/2) * Q with Q = sum_i F_i (theta_i - theta'_i)^2, so asking for
            # L_reg/L_task = r at the current drift gives lambda = 2 r L_task / Q. This
            # only puts lambda in the right decade -- Q keeps growing as training goes on.
            if args.lambda_auto > 0 and anchor and global_step == args.lambda_auto_steps:
                quadratic = consolidation_scale(model, anchor, fisher)
                task_loss = (totals["l1"] + totals["l2"] + totals["l3"]) / (i + 1)
                if quadratic > 0:
                    lambda_reg = 2.0 * args.lambda_auto * task_loss / quadratic
                    print(f"[lambda-auto] step {global_step}: task {task_loss:.5f} | "
                          f"sum F (theta-theta')^2 = {quadratic:.6e} -> lambda = {lambda_reg:.6e}",
                          flush=True)
                else:
                    print(f"[lambda-auto] step {global_step}: no drift yet, lambda left at 0.")

            step = i * normal_loader.batch_size * 2
            if args.eval_steps > 0 and step % args.eval_steps == 0 and step != 0:
                task_loss = (totals["l1"] + totals["l2"] + totals["l3"]) / (i + 1)
                reg_loss = totals["reg"] / (i + 1)
                print(f"epoch {epoch + 1} | step {step} | l1 {totals['l1'] / (i + 1):.5f} "
                      f"| l2 {totals['l2'] / (i + 1):.5f} | l3 {totals['l3'] / (i + 1):.5f} "
                      f"| reg {reg_loss:.5f} | reg/task {reg_loss / max(task_loss, 1e-8):.4f} "
                      f"| drift {mean_absolute_drift(model, anchor):.3e}", flush=True)
                best_auc = run_evaluation(
                    model, testloader, args, prompt_text, gt, gtsegments, gtlabels,
                    video_meta, device, epoch, global_step, best_auc, optimizer, lambda_reg
                )
                model.train()

            if args.debug_max_steps and global_step >= args.debug_max_steps:
                print(f"Stopped after {global_step} steps (--debug-max-steps).")
                return

        scheduler.step()
        epoch_task = (totals["l1"] + totals["l2"] + totals["l3"]) / max(1, num_steps)
        epoch_reg = totals["reg"] / max(1, num_steps)
        # reg/task is the number to calibrate lambda against: aim for roughly 0.05-0.30.
        # Below that the anchor has no pull; above it the model is frozen and stage 2 is moot.
        print(f"=== end of epoch {epoch + 1} | l1 {totals['l1'] / max(1, num_steps):.5f} "
              f"| l2 {totals['l2'] / max(1, num_steps):.5f} "
              f"| reg {epoch_reg:.5f} | reg/task {epoch_reg / max(epoch_task, 1e-8):.4f} "
              f"| drift {mean_absolute_drift(model, anchor):.3e} ===", flush=True)

        torch.save(model.state_dict(), args.save_cur_path)
        epoch_path = Path(args.epoch_checkpoint_dir) / f"model_epoch_{epoch + 1:03d}_rescale_ewc.pth"
        torch.save(model.state_dict(), epoch_path)

        # Under --select-metric none the last epoch's evaluation IS the model that gets
        # saved, so mark that row: the comparison table can then read one row per run.
        best_auc = run_evaluation(
            model, testloader, args, prompt_text, gt, gtsegments, gtlabels,
            video_meta, device, epoch, global_step, best_auc, optimizer, lambda_reg,
            is_final=(args.select_metric == "none" and epoch == args.max_epoch - 1),
        )

        # Rewinding to the best-so-far checkpoint only makes sense when a metric is
        # actually selecting one. Under --select-metric none the run just keeps going.
        if args.select_metric != "none" and Path(args.checkpoint_path).exists():
            checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])

    if args.select_metric != "none" and Path(args.checkpoint_path).exists():
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        torch.save(checkpoint["model_state_dict"], args.output_model_path)
        print(f"Saved best-{args.select_metric} weights:", args.output_model_path)
    else:
        torch.save(model.state_dict(), args.output_model_path)
        print("Saved final-epoch weights:", args.output_model_path)


def run_evaluation(model, testloader, args, prompt_text, gt, gtsegments, gtlabels,
                   video_meta, device, epoch, global_step, best_auc, optimizer, lambda_reg=0.0,
                   is_final=False):
    metrics, _, _ = evaluate(
        model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels,
        video_meta, args.target_classes, device,
    )
    print(format_summary(metrics, args.target_classes), flush=True)

    if args.metrics_csv:
        append_metrics_csv(args.metrics_csv, {
            "run": args.run_tag, "epoch": epoch + 1, "step": global_step,
            "mu": args.mu, "rescale_mode": args.rescale_mode,
            "class_weight_file": Path(args.class_weight_file).name if args.class_weight_file else "",
            # From the weight file, not from --adaptive-beta: the run used whatever beta
            # produced that file, and the flag on this command line may say something else.
            "adaptive_beta": getattr(args, "adaptive_beta_used", ""),
            "regularizer": args.regularizer, "lambda_reg": args.lambda_reg,
            "lambda_used": round(lambda_reg, 8), "is_final": int(is_final),
            **{key: round(value, 4) for key, value in metrics.items()},
        })

    if args.select_metric == "none":
        return best_auc

    score = metrics[args.select_metric]
    if score > best_auc:
        best_auc = float(score)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "ap": best_auc,
        }, args.checkpoint_path)
        print(f"  new best {args.select_metric} {best_auc:.2f} -> {args.checkpoint_path}")
    return best_auc


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    from model import CLIPVAD

    from dataset_rescale import UCFRescaleDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option_rescale.parser.parse_args()

    # Checked before the model is built, which takes long enough that a typo found later
    # costs a real wait -- and long enough that a silent fallback to no rescaling at all
    # would be easy to miss in a log.
    if args.rescale_mode == "adaptive_class":
        if not args.class_weight_file:
            raise SystemExit(
                "--rescale-mode adaptive_class needs --class-weight-file. Produce one with:\n"
                "  python ucf_train_difficulty.py --pretrained-model-path <theta'.pth> "
                "--difficulty-output model/adaptive_weights.json"
            )
        if not Path(args.class_weight_file).exists():
            raise SystemExit(f"--class-weight-file {args.class_weight_file} does not exist.")

    setup_seed(args.seed)

    label_map = ucf_option_rescale.UCF_LABEL_MAP
    target_classes = set(args.target_classes)

    pin_memory = bool(args.pin_memory and device == "cuda")
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": pin_memory}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    normal_dataset = UCFRescaleDataset(args.visual_length, args.train_list, False, args.feature_root, normal=True)
    anomaly_dataset = UCFRescaleDataset(args.visual_length, args.train_list, False, args.feature_root, normal=False)

    sampler = build_anomaly_sampler(anomaly_dataset, target_classes, args.target_oversample)
    if sampler is not None:
        print(f"Target-class oversampling active: weight {args.target_oversample}")

    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True,
                               drop_last=True, **loader_kwargs)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size,
                                shuffle=(sampler is None), sampler=sampler,
                                drop_last=True, **loader_kwargs)

    test_dataset = UCFRescaleDataset(args.visual_length, args.test_list, True, args.feature_root)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **loader_kwargs)

    model = CLIPVAD(
        args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
        args.visual_head, args.visual_layers, args.attn_window,
        args.prompt_prefix, args.prompt_postfix, device,
    )
    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)


if __name__ == "__main__":
    main()
