"""Temporal shift-consistency fine-tuning for VadCLIP on UCF-Crime.

The model itself is untouched: this script feeds each video twice (a full view and a
temporally shifted view) through a single ``CLIPVAD.forward`` call by stacking the two
views along the batch dimension, keeps the three original losses on the full view only,
and adds a consistency term that asks the two views to agree on the overlapping frames.

``model`` and ``ucf_test_description`` are imported lazily so that
``shift_consistency_loss`` can be unit tested without pulling in the CLIP dependencies.
"""

import os

# Must be set before torch initialises cuBLAS, otherwise --deterministic cannot make
# matmul reductions reproducible. Harmless when --deterministic is off. Same line, same
# reason, as baseline/src/ucf_train.py.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import ucf_option_augment
from utils.dataset_augment import UCFAugmentDataset
from utils.tools import get_batch_label, get_prompt_text


# --- Original VadCLIP losses, copied verbatim from ucf_train.py. Do not modify. -------

def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss


def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss


# --- Shift-consistency loss ----------------------------------------------------------

def _to_long_vector(value, batch_size, device):
    """Accept an int or a per-sample tensor and return a ``[batch_size]`` long tensor."""
    if torch.is_tensor(value):
        vector = value.detach().to(device=device, dtype=torch.long).reshape(-1)
        if vector.numel() == 1 and batch_size > 1:
            vector = vector.repeat(batch_size)
        return vector
    return torch.full((batch_size,), int(value), device=device, dtype=torch.long)


def overlap_sizes(len_full, offset, grid_length, batch_size=None, device="cpu"):
    """Number of overlapping positions per video, for a signed offset.

    The overlap in full-view coordinates is ``0 <= p < len_full`` intersected with
    ``0 <= p - offset < grid_length``, which collapses to
    ``min(len_full, grid_length + offset) - max(0, offset)``.
    """
    if batch_size is None:
        batch_size = len(len_full) if hasattr(len_full, "__len__") else 1
    lengths = _to_long_vector(len_full, batch_size, device)
    offsets = _to_long_vector(offset, batch_size, device)
    upper = torch.minimum(lengths, offsets + int(grid_length))
    return (upper - offsets.clamp(min=0)).clamp(min=0)


def count_without_overlap(len_full, offset, batch_size=None, device="cpu", grid_length=256):
    """Videos left with no overlap after the shift; they contribute nothing to the loss.

    Only positive (head) offsets can empty the overlap, by cutting away more than the
    video has. A negative offset merely repositions the content, so the ``tail`` and
    ``both`` directions cannot drop a video this way.
    """
    return int((overlap_sizes(len_full, offset, grid_length, batch_size, device) <= 0).sum().item())


def shift_consistency_loss(logits_full, logits_shift, len_full, offset,
                           branch="c", detach_anchor=False):
    """
    logits_full  : C-branch [B, T, 1]  or A-branch [B, T, 14]
    logits_shift : same shape, for the shifted view
    len_full     : [B] valid length of the full view
    offset       : signed int, or [B] tensor when the offset is sampled per item
    branch       : 'c' -> MSE on sigmoid; 'a' -> symmetric KL on the class softmax

    Position ``p`` of the full view matches position ``p - offset`` of the shifted view.
    The overlap is the set of ``p`` for which both ends exist: ``p < len_full`` and
    ``0 <= p - offset < T``. For a positive (head) offset that is ``[offset, len_full)``;
    for a negative (tail) offset it is ``[0, min(len_full, T - |offset|))``. Videos left
    with an empty overlap are dropped from the average.
    """
    batch_size, length, channels = logits_full.shape
    device = logits_full.device

    offsets = _to_long_vector(offset, batch_size, device).clamp(min=-(length - 1), max=length)
    lengths = _to_long_vector(len_full, batch_size, device).clamp(min=0, max=length)

    positions = torch.arange(length, device=device).unsqueeze(0)

    # Align the shifted view back onto the full view's time axis. The shifted view holds
    # full-view content ``p`` at index ``p - offset``, so a position of the full view has
    # a counterpart exactly when that index lands inside the grid. One expression covers
    # both signs; for offset >= 0 it reduces to the old ``offset <= p < len_full``.
    source_index = positions - offsets.unsqueeze(1)
    valid = (positions < lengths.unsqueeze(1)) & (source_index >= 0) & (source_index < length)

    gather_index = source_index.clamp(min=0, max=length - 1).unsqueeze(-1)
    gather_index = gather_index.expand(batch_size, length, channels)
    aligned_shift = torch.gather(logits_shift, 1, gather_index)

    anchor = logits_full.detach() if detach_anchor else logits_full

    if branch == "c":
        probability_full = torch.sigmoid(anchor.float())
        probability_shift = torch.sigmoid(aligned_shift.float())
        elementwise = (probability_full - probability_shift).pow(2).mean(dim=-1)
    elif branch == "a":
        log_p = F.log_softmax(anchor.float(), dim=-1)
        log_q = F.log_softmax(aligned_shift.float(), dim=-1)
        kl_pq = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
        kl_qp = (log_q.exp() * (log_q - log_p)).sum(dim=-1)
        elementwise = 0.5 * (kl_pq + kl_qp)
    else:
        raise ValueError(f"branch must be 'c' or 'a'. Got {branch!r}.")

    valid_weight = valid.to(elementwise.dtype)
    per_video_sum = (elementwise * valid_weight).sum(dim=1)
    per_video_count = valid_weight.sum(dim=1)
    has_overlap = per_video_count > 0
    if not bool(has_overlap.any()):
        return torch.zeros((), device=device, dtype=elementwise.dtype)
    return (per_video_sum[has_overlap] / per_video_count[has_overlap]).mean()


def append_metrics_csv(path, row):
    """Append one row, reusing an existing header so a whole sweep shares one file."""
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


def load_checkpoint_dict(path, map_location=None):
    """Load a checkpoint written by this script.

    These files hold optimizer state and a scalar score alongside the weights, so
    weights_only=True (the PyTorch 2.6+ default) refuses to unpickle them.
    """
    return torch.load(path, map_location=map_location, weights_only=False)


def load_model_weights(model, path, device):
    """Accept either a bare state_dict or a full ucf_train.py-style checkpoint dict."""
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)


def combined_consistency_loss(logits1_full, logits1_shift, logits2_full, logits2_shift,
                              len_full, offset, branch, detach_anchor):
    """Dispatch on ``--consistency-branch``; 'both' sums the C- and A-branch terms."""
    if branch in ("c", "both"):
        loss_c = shift_consistency_loss(
            logits1_full, logits1_shift, len_full, offset, "c", detach_anchor
        )
    else:
        loss_c = torch.zeros((), device=logits1_full.device)

    if branch in ("a", "both"):
        loss_a = shift_consistency_loss(
            logits2_full, logits2_shift, len_full, offset, "a", detach_anchor
        )
    else:
        loss_a = torch.zeros((), device=logits2_full.device)

    return loss_c + loss_a, loss_c, loss_a


# --- Training ------------------------------------------------------------------------

def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    from ucf_test_description import test  # lazy: keeps the loss importable without CLIP

    model.to(device)

    # The single-view shortcut is only sound while the consistency term is identically
    # zero. Checking it here, before any GPU time is spent, beats discovering after ten
    # epochs that lambda was silently ignored.
    single_view = bool(args.skip_shifted_view)
    if single_view and (args.lambda_consistency != 0 or args.lambda_auto != 0):
        raise ValueError(
            "--skip-shifted-view drops the shifted view entirely, so the consistency term "
            "cannot be computed. It is only legal for the lambda-0 control. Got "
            f"--lambda-consistency {args.lambda_consistency} and --lambda-auto {args.lambda_auto}."
        )

    Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_cur_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.epoch_checkpoint_dir).mkdir(parents=True, exist_ok=True)

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    if args.use_pretrained_model:
        load_model_weights(model, args.pretrained_model_path, device)
        print("Loaded pretrained model:", args.pretrained_model_path)
    else:
        print("Training VadCLIP-specific layers from scratch.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    global_step = 0

    print(
        "Shift-consistency config:",
        "| lambda_consistency:", args.lambda_consistency,
        "| lambda_auto:", args.lambda_auto,
        "| lambda_auto_recalibrate:", args.lambda_auto_recalibrate,
        "| shift_offset:", args.shift_offset,
        "| shift_ratio:", args.shift_ratio,
        "| shift_direction:", args.shift_direction,
        "| random_shift:", args.random_shift,
        "| ratio_warmup:", args.shift_ratio_warmup,
        "| branch:", args.consistency_branch,
        "| detach_anchor:", args.consistency_detach,
        "| skip_shifted_view:", args.skip_shifted_view,
        "| deterministic:", args.deterministic,
        "| warmup_epochs:", args.consistency_warmup,
        "| select_metric:", args.select_metric,
        "| lr:", args.lr,
        "| from_pretrained:", args.use_pretrained_model,
    )

    # lambda_base is what the epoch ramp multiplies. --lambda-auto starts it at 0 and
    # solves for it once the two sides of the objective have been measured.
    lambda_base = args.lambda_consistency
    if args.lambda_auto > 0:
        lambda_base = 0.0
        print(f"Lambda auto-calibration on: after {args.lambda_auto_steps} steps with the "
              f"term switched off, lambda will be solved so that "
              f"lambda*L_consistency / L_task = {args.lambda_auto}.")
        if args.lambda_auto_recalibrate:
            print(f"Recalibration on: lambda is re-solved at the end of every epoch, "
                  f"moving by at most a factor of {args.lambda_auto_max_growth} each time "
                  f"({'no bound' if args.lambda_auto_max_growth <= 0 else 'bounded'}).")

    for e in range(args.max_epoch):
        model.train()
        loss_total1 = 0
        loss_total2 = 0
        loss_total3 = 0
        loss_total4 = 0
        no_overlap_total = 0
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        num_steps = min(len(normal_loader), len(anomaly_loader))
        lam = 0.0

        # Ratio curriculum: weak shifts first, full strength from epoch shift_ratio_warmup
        # on. RandAugment's finding is that the best augmentation magnitude rises as
        # training proceeds, so a constant magnitude is wrong at one end or the other.
        if args.shift_ratio_warmup > 0:
            shift_scale = min(1.0, (e + 1) / args.shift_ratio_warmup)
            for loader in (normal_loader, anomaly_loader):
                loader.dataset.set_shift_scale(shift_scale)
            print(f"epoch {e + 1} | shift_scale: {shift_scale:.3f}")

        for i in range(num_steps):
            normal_full, normal_shift, normal_label, normal_len, normal_len_shift, normal_offset = next(normal_iter)
            anomaly_full, anomaly_shift, anomaly_label, anomaly_len, anomaly_len_shift, anomaly_offset = next(anomaly_iter)

            visual_full = torch.cat([normal_full, anomaly_full], dim=0).to(device)
            text_labels_raw = list(normal_label) + list(anomaly_label)
            len_full = torch.cat([normal_len, anomaly_len], dim=0).to(device)
            offsets = torch.cat([normal_offset, anomaly_offset], dim=0).to(device)
            if single_view:
                # Nothing downstream reads them, and the visual one is ~33 MB a step.
                visual_shift, len_shift = None, None
            else:
                visual_shift = torch.cat([normal_shift, anomaly_shift], dim=0).to(device)
                len_shift = torch.cat([normal_len_shift, anomaly_len_shift], dim=0).to(device)
            text_labels = get_batch_label(text_labels_raw, prompt_text, label_map).to(device)

            batch = visual_full.shape[0]
            if single_view:
                # lambda is identically 0, so the shifted view's only effect would be to
                # double the visual batch and then be multiplied away. CLIPVAD is batch
                # independent (tests/test_two_view_batching.py), so the full view's logits
                # are the same as they would be inside the stacked batch.
                text_features, logits1_full, logits2_full = model(
                    visual_full, None, prompt_text, len_full
                )
                logits1_shift, logits2_shift = None, None
            else:
                # Both views in one forward pass; encoding the 14 text prompts twice is wasteful.
                visual_2view = torch.cat([visual_full, visual_shift], dim=0)
                lengths_2view = torch.cat([len_full, len_shift.clamp(min=1)], dim=0)
                text_features, logits1, logits2 = model(visual_2view, None, prompt_text, lengths_2view)

                logits1_full, logits1_shift = logits1[:batch], logits1[batch:]
                logits2_full, logits2_shift = logits2[:batch], logits2[batch:]

            # The three original losses see the full view only, so labels stay clean.
            loss1 = CLAS2(logits1_full, text_labels, len_full, device)
            loss2 = CLASM(logits2_full, text_labels, len_full, device)

            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 13 * 1e-1

            if single_view:
                loss4 = torch.zeros((), device=device)
            else:
                loss4, _, _ = combined_consistency_loss(
                    logits1_full, logits1_shift, logits2_full, logits2_shift,
                    len_full, offsets, args.consistency_branch, args.consistency_detach,
                )

            lam = lambda_base * min(1.0, (e + 1) / max(1, args.consistency_warmup))
            loss = loss1 + loss2 + loss3 + lam * loss4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total1 += loss1.item()
            loss_total2 += loss2.item()
            loss_total3 += loss3.item()
            loss_total4 += loss4.item()
            if not single_view:
                no_overlap_total += count_without_overlap(
                    len_full, offsets, batch, device, grid_length=args.visual_length
                )

            if global_step == 0:
                print(
                    f"[step 0] loss1={loss1.item():.8f} loss2={loss2.item():.8f} "
                    f"loss3={loss3.item():.8f} loss4_raw={loss4.item():.8f}"
                )
            global_step += 1

            # Solve for lambda instead of hand-searching it. The consistency term enters
            # as lambda * L4, so asking for a share r of the task loss gives
            # lambda = r * L_task / L4. Measuring both sides on the same steps makes the
            # setting transfer across shift configurations, which an absolute lambda does
            # not: changing the cut changes L4's scale.
            if args.lambda_auto > 0 and global_step == args.lambda_auto_steps:
                task_loss = (loss_total1 + loss_total2 + loss_total3) / (i + 1)
                consistency_loss = loss_total4 / (i + 1)
                if consistency_loss > 0:
                    lambda_base = args.lambda_auto * task_loss / consistency_loss
                    print(f"[lambda-auto] step {global_step}: task {task_loss:.6f} | "
                          f"consistency {consistency_loss:.6e} -> lambda = {lambda_base:.6e}",
                          flush=True)
                else:
                    print(f"[lambda-auto] step {global_step}: consistency term is 0, "
                          f"lambda left at 0.", flush=True)

            step = i * normal_loader.batch_size * 2
            if args.eval_steps > 0 and step % args.eval_steps == 0 and step != 0:
                print(
                    "epoch:", e + 1,
                    "| step:", step,
                    "| loss1:", loss_total1 / (i + 1),
                    "| loss2:", loss_total2 / (i + 1),
                    "| loss3:", loss_total3 / (i + 1),
                    "| loss4_raw:", loss_total4 / (i + 1),
                    "| lambda:", lam,
                )
                AUC, AP = test(model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
                model.train()
                log_evaluation(args, AUC, AP, e, global_step, lam, is_final=False)
                if args.select_metric != "none" and AUC > ap_best:
                    ap_best = float(AUC)  # float, not a numpy scalar, so the file stays plain
                    torch.save(
                        {
                            "epoch": e,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "ap": ap_best,
                        },
                        args.checkpoint_path,
                    )

            if args.debug_max_steps and global_step >= args.debug_max_steps:
                print(f"Stopped after {global_step} steps (--debug-max-steps).")
                return

        scheduler.step()
        print(
            "epoch:", e + 1,
            "| loss1:", loss_total1 / max(1, num_steps),
            "| loss2:", loss_total2 / max(1, num_steps),
            "| loss3:", loss_total3 / max(1, num_steps),
            "| loss4_raw:", loss_total4 / max(1, num_steps),
            "| videos_without_overlap:", no_overlap_total,
        )

        # Re-solve lambda once per epoch rather than once per run. The one-shot solve
        # pins lambda to the task loss of an almost-untrained model, and both sides of
        # the ratio then move: over round 2 the task loss fell about five-fold while the
        # consistency loss fell ten to forty-fold, so the share the term actually held
        # drifted to roughly a third of the target at the top end. Three runs meant to
        # span a decade of dose ended up spanning a factor of two, which is why that
        # lambda curve could not separate anything.
        #
        # The growth cap is not decoration. Holding a fixed share against a consistency
        # loss that is being driven towards zero asks for an unbounded lambda, so this is
        # a feedback loop with no fixed point; the cap is what makes it a controller
        # instead of a divergence.
        if (args.lambda_auto > 0 and args.lambda_auto_recalibrate
                and lambda_base > 0 and e < args.max_epoch - 1):
            task_epoch = (loss_total1 + loss_total2 + loss_total3) / max(1, num_steps)
            cons_epoch = loss_total4 / max(1, num_steps)
            if cons_epoch > 0 and task_epoch > 0:
                solved = args.lambda_auto * task_epoch / cons_epoch
                capped = solved
                if args.lambda_auto_max_growth > 0:
                    capped = min(max(solved, lambda_base / args.lambda_auto_max_growth),
                                 lambda_base * args.lambda_auto_max_growth)
                note = "" if capped == solved else f" (capped from {solved:.6e})"
                print(f"[lambda-auto] end of epoch {e + 1}: realised share "
                      f"{lam * cons_epoch / task_epoch:.4f} vs target {args.lambda_auto} "
                      f"| lambda {lambda_base:.6e} -> {capped:.6e}{note}", flush=True)
                lambda_base = capped
            else:
                print(f"[lambda-auto] end of epoch {e + 1}: consistency term is 0, "
                      f"lambda left at {lambda_base:.6e}.", flush=True)

        torch.save(model.state_dict(), args.save_cur_path)
        epoch_checkpoint_path = Path(args.epoch_checkpoint_dir) / f"model_epoch_{e + 1:03d}_shift_consistency.pth"
        torch.save(model.state_dict(), epoch_checkpoint_path)
        print("Saved epoch checkpoint:", epoch_checkpoint_path)

        # An epoch-end score, so a sweep gets one comparable row per epoch even with
        # --eval-steps 0. It only logs; it never updates the best checkpoint, so runs made
        # before this option existed keep selecting exactly the same weights.
        if testloader is not None and (args.metrics_csv or args.select_metric == "none"):
            AUC, AP = test(model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
            log_evaluation(args, AUC, AP, e, global_step, lam,
                           is_final=(e == args.max_epoch - 1))

        # Same best-checkpoint reload as ucf_train.py, guarded for the case where no
        # evaluation has run yet (--eval-steps 0 or a very short schedule). Under
        # --select-metric none there is no selection to reload.
        if args.select_metric != "none" and Path(args.checkpoint_path).exists():
            checkpoint = load_checkpoint_dict(args.checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])

    if args.select_metric != "none" and Path(args.checkpoint_path).exists():
        checkpoint = load_checkpoint_dict(args.checkpoint_path, map_location=device)
        torch.save(checkpoint["model_state_dict"], args.output_model_path)
        print(f"Saved best-{args.select_metric} weights:", args.output_model_path)
    else:
        torch.save(model.state_dict(), args.output_model_path)
        print("Saved final-epoch weights:", args.output_model_path)


def log_evaluation(args, auc, ap, epoch, global_step, lam, is_final):
    """One row per evaluation, so a sweep is readable from a single CSV."""
    if not args.metrics_csv:
        return
    append_metrics_csv(args.metrics_csv, {
        "run": args.run_tag,
        "epoch": epoch + 1,
        "step": global_step,
        "lambda_consistency": args.lambda_consistency,
        "lambda_used": round(float(lam), 10),
        "shift_offset": args.shift_offset,
        "shift_ratio": args.shift_ratio,
        "shift_direction": args.shift_direction,
        "random_shift": int(bool(args.random_shift)),
        "detach": int(bool(args.consistency_detach)),
        "branch": args.consistency_branch,
        "seed": args.seed,
        "is_final": int(is_final),
        "auc": round(float(auc), 4),
        "ap": round(float(ap), 4),
    })


def setup_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Seeding alone does not make a GPU run reproducible: reduction order is free to
    # change between runs. baseline/src/ucf_train.py pins it behind the same flag, and the
    # control run has to be comparable with runs made by that script.
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("Deterministic mode ON (cudnn.deterministic, no benchmark, "
              "CUBLAS_WORKSPACE_CONFIG=" + os.environ.get("CUBLAS_WORKSPACE_CONFIG", "unset") + ")")


if __name__ == "__main__":
    from model import CLIPVAD  # lazy: keeps the loss importable without CLIP
    from ucf_train_class_prototype import UCFPrototypeDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option_augment.parser.parse_args()
    setup_seed(args.seed, args.deterministic)

    label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})

    pin_memory = bool(args.pin_memory and device == "cuda")
    dataloader_kwargs = {"num_workers": args.num_workers, "pin_memory": pin_memory}
    # A persistent worker pool keeps its own pickled copy of the dataset, so the per-epoch
    # set_shift_scale call would never reach it. Non-persistent workers are re-forked at
    # every iter(), which is exactly what the curriculum needs.
    if args.num_workers > 0 and args.shift_ratio_warmup <= 0:
        dataloader_kwargs["persistent_workers"] = True

    cut_kwargs = dict(
        shift_offset=args.shift_offset,
        random_shift=args.random_shift,
        shift_ratio=args.shift_ratio,
        shift_direction=args.shift_direction,
    )

    normal_dataset = UCFAugmentDataset(
        args.visual_length, args.train_list, False, label_map, args.feature_root,
        normal=True, **cut_kwargs,
    )
    normal_loader = DataLoader(
        normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **dataloader_kwargs
    )
    anomaly_dataset = UCFAugmentDataset(
        args.visual_length, args.train_list, False, label_map, args.feature_root,
        normal=False, **cut_kwargs,
    )
    anomaly_loader = DataLoader(
        anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **dataloader_kwargs
    )

    # The test loader keeps the plain VadCLIP evaluation path: no shifted view. It reuses
    # UCFPrototypeDataset because ucf_test_description.test expects (feat, label, length, id).
    test_dataset = UCFPrototypeDataset(args.visual_length, args.test_list, True, args.feature_root, False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **dataloader_kwargs)

    model = CLIPVAD(
        args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
        args.visual_head, args.visual_layers, args.attn_window,
        args.prompt_prefix, args.prompt_postfix, device,
    )

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)
