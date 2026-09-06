"""End-to-end smoke test of the shift-consistency training loop.

Runs one tiny epoch on synthetic features with a stubbed CLIP encoder, so integration
bugs (tuple unpacking, length dtypes, checkpoint writing) surface here instead of after
a Colab session has already started.

Run from VadCLIP/src:  python tests/test_train_smoke.py
"""

import contextlib
import csv
import io
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from test_two_view_batching import install_clip_stub

install_clip_stub()

import ucf_option_augment  # noqa: E402
from model import CLIPVAD  # noqa: E402
from ucf_train_augment import (  # noqa: E402
    combined_consistency_loss,
    load_checkpoint_dict,
    load_model_weights,
    train,
)
from utils.dataset_augment import UCFAugmentDataset  # noqa: E402

VISUAL_LENGTH = 64  # smaller grid keeps the smoke test fast
FEATURE_WIDTH = 512
LABEL_MAP = {"Normal": "normal", "Abuse": "abuse", "Arson": "arson"}


def build_fixture(root):
    rows = []
    rng = np.random.default_rng(0)
    plan = [("Normal", 16, 90), ("Abuse", 8, 70), ("Arson", 8, 20)]
    for label, count, snippets in plan:
        (root / label).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            video_id = f"{label}{index:03d}_x264"
            relative = f"{label}/{video_id}__0.npy"
            np.save(root / relative, rng.standard_normal((snippets, FEATURE_WIDTH), dtype=np.float32))
            rows.append({"path": relative, "label": label, "video_id": video_id})

    list_path = root / "train_list.csv"
    with open(list_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "video_id"])
        writer.writeheader()
        writer.writerows(rows)

    # train() loads these up front; evaluation never fires with --eval-steps 0.
    np.save(root / "gt.npy", np.zeros(64, dtype=np.float32))
    np.save(root / "gt_segment.npy", np.array([[0, 1]], dtype=np.float32))
    np.save(root / "gt_label.npy", np.array(["Abuse"], dtype=object), allow_pickle=True)
    return list_path


def build_args(root, list_path, **overrides):
    args = ucf_option_augment.parser.parse_args([])
    args.visual_length = VISUAL_LENGTH
    args.classes_num = len(LABEL_MAP)
    args.batch_size = 4
    args.max_epoch = 1
    args.eval_steps = 0
    args.shift_offset = 8
    args.num_workers = 0
    args.pin_memory = False
    args.use_pretrained_model = False
    args.feature_root = str(root)
    args.train_list = str(list_path)
    args.test_list = str(list_path)
    args.gt_path = str(root / "gt.npy")
    args.gt_segment_path = str(root / "gt_segment.npy")
    args.gt_label_path = str(root / "gt_label.npy")
    args.checkpoint_path = str(root / "model" / "checkpoint.pth")
    args.output_model_path = str(root / "model" / "output.pth")
    args.save_cur_path = str(root / "model" / "cur.pth")
    args.epoch_checkpoint_dir = str(root / "model" / "epochs")
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def build_model(args):
    torch.manual_seed(234)
    return CLIPVAD(
        args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
        args.visual_head, args.visual_layers, args.attn_window,
        args.prompt_prefix, args.prompt_postfix, "cpu",
    )


def build_loaders(args):
    cut = dict(
        shift_offset=args.shift_offset,
        random_shift=args.random_shift,
        shift_ratio=args.shift_ratio,
        shift_direction=args.shift_direction,
    )
    normal_dataset = UCFAugmentDataset(
        args.visual_length, args.train_list, False, LABEL_MAP, args.feature_root,
        normal=True, **cut,
    )
    anomaly_dataset = UCFAugmentDataset(
        args.visual_length, args.train_list, False, LABEL_MAP, args.feature_root,
        normal=False, **cut,
    )
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    return normal_loader, anomaly_loader


def test_one_epoch_runs(root, list_path, branch, random_shift):
    args = build_args(root, list_path, consistency_branch=branch, random_shift=random_shift)
    model = build_model(args)
    normal_loader, anomaly_loader = build_loaders(args)

    train(model, normal_loader, anomaly_loader, None, args, LABEL_MAP, "cpu")

    epoch_checkpoints = sorted(Path(args.epoch_checkpoint_dir).glob("*.pth"))
    assert epoch_checkpoints, "no epoch checkpoint was written"
    assert Path(args.output_model_path).exists(), "final model was not saved"
    # ucf_analyze_checkpoints.py sorts on the last digit group in the stem.
    assert epoch_checkpoints[0].stem == "model_epoch_001_shift_consistency"
    print(f"  ok: one epoch ran (branch={branch}, random_shift={random_shift}), checkpoints written")


def test_consistency_gradient_reaches_the_adapter(root, list_path):
    """The consistency term alone must produce gradient in the temporal adapter."""
    args = build_args(root, list_path)
    model = build_model(args)
    normal_loader, _ = build_loaders(args)

    features_full, features_shift, _, len_full, len_shift, offsets = next(iter(normal_loader))
    batch = features_full.shape[0]
    visual = torch.cat([features_full, features_shift], dim=0)
    lengths = torch.cat([len_full, len_shift.clamp(min=1)], dim=0)

    prompt_text = list(LABEL_MAP.values())
    _, logits1, logits2 = model(visual, None, prompt_text, lengths)
    loss4, loss_c, loss_a = combined_consistency_loss(
        logits1[:batch], logits1[batch:], logits2[:batch], logits2[batch:],
        len_full, offsets, "both", False,
    )
    assert loss4.item() > 0, "an untrained model should not already be shift invariant"

    model.zero_grad()
    loss4.backward()

    adapter_grad = model.temporal.resblocks[0].attn.in_proj_weight.grad
    position_grad = model.frame_position_embeddings.weight.grad
    assert adapter_grad is not None and torch.any(adapter_grad != 0), "no gradient in the temporal adapter"
    assert position_grad is not None and torch.any(position_grad != 0), "no gradient in the position embeddings"
    assert all(p.grad is None or torch.all(p.grad == 0) for p in model.clipmodel.parameters()), \
        "frozen CLIP must not receive gradient"
    print(f"  ok: consistency gradient reaches the adapter and position embeddings "
          f"(loss_c={loss_c.item():.4f}, loss_a={loss_a.item():.4f})")


def test_checkpoint_dict_roundtrip(root):
    """Regression: PyTorch 2.6 defaults torch.load to weights_only=True.

    The checkpoint written mid-epoch carries optimizer state and a score, and the score
    used to be a numpy scalar straight from roc_auc_score, which weights_only=True
    refuses to unpickle. Both the current format and older files must load.
    """
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score

    module = nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-5)
    score = roc_auc_score([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])

    current = root / "ckpt_current.pth"
    torch.save({"epoch": 0, "model_state_dict": module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "ap": float(score)}, current)
    loaded = load_checkpoint_dict(current)
    assert "model_state_dict" in loaded and isinstance(loaded["ap"], float)

    # A file written before the float() cast, i.e. with a numpy scalar inside.
    legacy = root / "ckpt_legacy.pth"
    torch.save({"epoch": 0, "model_state_dict": module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "ap": np.float64(0.8802)}, legacy)

    # The hazard is real on every torch >= 1.13, not only where it is the default: strict
    # loading must reject this file, and our loader must still read it.
    strict_rejected = False
    try:
        torch.load(legacy, weights_only=True)
    except Exception:
        strict_rejected = True
    assert strict_rejected, "expected weights_only=True to reject a numpy scalar in the checkpoint"

    loaded_legacy = load_checkpoint_dict(legacy)
    assert abs(float(loaded_legacy["ap"]) - 0.8802) < 1e-9

    # load_model_weights must accept both a checkpoint dict and a bare state_dict.
    fresh = nn.Linear(4, 2)
    load_model_weights(fresh, current, "cpu")
    bare = root / "bare_state.pth"
    torch.save(module.state_dict(), bare)
    load_model_weights(fresh, bare, "cpu")
    print("  ok: checkpoint dicts load under PyTorch 2.6+ (current and legacy numpy scalar)")


def test_best_checkpoint_reload_path(root, list_path):
    """Run with evaluation enabled so the mid-epoch save and end-of-epoch reload both fire."""
    args = build_args(root, list_path, eval_steps=8, batch_size=4)

    def fake_test(model, loader, maxlen, prompt_text, gt, gtsegments, gtlabels, device):
        fake_test.calls += 1
        return 0.5 + 0.1 * fake_test.calls, 0.3  # improving AUC forces a checkpoint save
    fake_test.calls = 0

    import ucf_test_description
    original = ucf_test_description.test
    ucf_test_description.test = fake_test
    try:
        model = build_model(args)
        normal_loader, anomaly_loader = build_loaders(args)
        train(model, normal_loader, anomaly_loader, None, args, LABEL_MAP, "cpu")
    finally:
        ucf_test_description.test = original

    assert fake_test.calls > 0, "evaluation never ran, the reload path was not exercised"
    assert Path(args.checkpoint_path).exists(), "best checkpoint was not written"
    reloaded = load_checkpoint_dict(args.checkpoint_path)
    assert isinstance(reloaded["ap"], float)
    assert Path(args.output_model_path).exists()
    print(f"  ok: mid-epoch save + end-of-epoch reload ran ({fake_test.calls} evaluations)")



def test_tuned_cut_and_auto_lambda(root, list_path):
    """The round-2 configuration end to end: ratio cut, both directions, random
    magnitude, a curriculum on the magnitude, lambda solved from its share of the
    objective, final weights kept instead of the best test score, one CSV row per epoch.
    """
    metrics_csv = root / "sweep_metrics.csv"
    args = build_args(
        root, list_path,
        max_epoch=2,
        eval_steps=0,
        shift_ratio=0.1,
        shift_offset=0,
        shift_direction="both",
        random_shift=True,
        shift_ratio_warmup=2,
        lambda_auto=0.1,
        lambda_auto_steps=2,
        consistency_warmup=1,
        select_metric="none",
        run_tag="tuned",
        metrics_csv=str(metrics_csv),
        # Its own path: earlier tests in this tmpdir already wrote the shared one.
        checkpoint_path=str(root / "model" / "checkpoint_tuned.pth"),
        output_model_path=str(root / "model" / "output_tuned.pth"),
    )

    def fake_test(model, loader, maxlen, prompt_text, gt, gtsegments, gtlabels, device):
        fake_test.calls += 1
        return 80.0 + fake_test.calls, 30.0
    fake_test.calls = 0

    import ucf_test_description
    original = ucf_test_description.test
    ucf_test_description.test = fake_test
    try:
        model = build_model(args)
        normal_loader, anomaly_loader = build_loaders(args)
        # A stand-in loader object: fake_test never touches it, but train() must see
        # something other than None for the epoch-end row to be written.
        train(model, normal_loader, anomaly_loader, normal_loader, args, LABEL_MAP, "cpu")
    finally:
        ucf_test_description.test = original

    with open(metrics_csv, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2, f"expected one row per epoch, got {len(rows)}"
    assert [row["epoch"] for row in rows] == ["1", "2"]
    assert rows[-1]["is_final"] == "1" and rows[0]["is_final"] == "0"
    assert rows[0]["run"] == "tuned"
    assert rows[0]["shift_direction"] == "both"
    assert float(rows[-1]["lambda_used"]) > 0, "lambda-auto never solved for a value"

    assert not Path(args.checkpoint_path).exists(), (
        "--select-metric none must not write a best-on-test checkpoint"
    )
    assert Path(args.output_model_path).exists(), "final weights were not saved"
    print(f"  ok: tuned cut + lambda-auto ran, lambda={rows[-1]['lambda_used']}, "
          f"final weights kept")

def _run_recalibration(root, list_path, tag, recalibrate, max_growth=1.5, epochs=3):
    """Three epochs of lambda-auto, returning the lambda used in each one."""
    metrics_csv = root / f"recal_{tag}.csv"
    args = build_args(
        root, list_path,
        max_epoch=epochs,
        eval_steps=0,
        lambda_auto=0.1,
        lambda_auto_steps=2,
        lambda_auto_recalibrate=recalibrate,
        lambda_auto_max_growth=max_growth,
        consistency_warmup=1,
        select_metric="none",
        run_tag=tag,
        metrics_csv=str(metrics_csv),
        checkpoint_path=str(root / "model" / f"checkpoint_{tag}.pth"),
        output_model_path=str(root / "model" / f"output_{tag}.pth"),
        save_cur_path=str(root / "model" / f"cur_{tag}.pth"),
        epoch_checkpoint_dir=str(root / "model" / f"epochs_{tag}"),
    )

    def fake_test(model, loader, maxlen, prompt_text, gt, gtsegments, gtlabels, device):
        return 80.0, 30.0

    import ucf_test_description
    original = ucf_test_description.test
    ucf_test_description.test = fake_test
    try:
        model = build_model(args)
        normal_loader, anomaly_loader = build_loaders(args)
        train(model, normal_loader, anomaly_loader, normal_loader, args, LABEL_MAP, "cpu")
    finally:
        ucf_test_description.test = original

    with open(metrics_csv, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == epochs, f"expected {epochs} rows, got {len(rows)}"
    return [float(row["lambda_used"]) for row in rows]


def test_lambda_recalibration_moves_and_is_capped(root, list_path):
    """Recalibration must change lambda between epochs, and never by more than the cap.

    The one-shot solve pins lambda to the task loss of an almost-untrained model. Round 2
    measured the cost of that: three runs asked for shares of 0.01 / 0.03 / 0.10 and
    ended at 1.9% / 2.6% / 4.0%, so the dose they were meant to separate by a decade
    separated by a factor of two instead. This test only checks the mechanism -- that the
    value moves, that it moves within its leash, and that the default is still the
    one-shot behaviour round 2 ran with.
    """
    max_growth = 1.5
    moving = _run_recalibration(root, list_path, "recal_on", True, max_growth=max_growth)
    fixed = _run_recalibration(root, list_path, "recal_off", False)

    assert all(value > 0 for value in moving), "lambda-auto never solved for a value"
    assert moving[1] != moving[0], "recalibration did not change lambda after epoch 1"

    for before, after in zip(moving, moving[1:]):
        ratio = after / before
        assert 1.0 / max_growth - 1e-9 <= ratio <= max_growth + 1e-9, (
            f"recalibration moved lambda by {ratio:.4f}x, outside the {max_growth}x cap"
        )

    assert len(set(fixed)) == 1, (
        f"without --lambda-auto-recalibrate lambda must stay put, saw {fixed}"
    )
    assert abs(fixed[0] - moving[0]) < 1e-9, (
        "the first epoch must be identical either way: recalibration only acts afterwards"
    )
    print(f"  ok: lambda recalibrates within its cap {moving[0]:.4e} -> "
          f"{' -> '.join(f'{v:.4e}' for v in moving[1:])}, and stays fixed when off")


def _step0_losses(text):
    """Pull the ``[step 0]`` line the trainer prints on its very first optimizer step."""
    match = re.search(
        r"\[step 0\] loss1=([\d.eE+-]+) loss2=([\d.eE+-]+) loss3=([\d.eE+-]+) "
        r"loss4_raw=([\d.eE+-]+)", text)
    assert match, "trainer never printed a step-0 line: " + text[:2000]
    return [float(g) for g in match.groups()]


def _run_lambda_zero(root, list_path, tag, skip_shifted_view):
    """One epoch of the lambda-0 control, with and without the shifted view."""
    args = build_args(
        root, list_path,
        max_epoch=1,
        eval_steps=0,
        lambda_consistency=0.0,
        lambda_auto=0.0,
        select_metric="none",
        skip_shifted_view=skip_shifted_view,
        checkpoint_path=str(root / "model" / f"checkpoint_{tag}.pth"),
        output_model_path=str(root / "model" / f"output_{tag}.pth"),
        save_cur_path=str(root / "model" / f"cur_{tag}.pth"),
        epoch_checkpoint_dir=str(root / "model" / f"epochs_{tag}"),
    )

    # Same seed, same construction order, so both variants see the same initial weights
    # and the same batch order. Anything left over is the change under test.
    torch.manual_seed(4321)
    np.random.seed(4321)
    model = build_model(args)
    normal_loader, anomaly_loader = build_loaders(args)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        train(model, normal_loader, anomaly_loader, None, args, LABEL_MAP, "cpu")
    return args, model, _step0_losses(buffer.getvalue())


def test_skip_shifted_view_matches_the_two_view_control(root, list_path):
    """--skip-shifted-view must not change the lambda-0 control, only its cost.

    The control forwards every video twice and then multiplies the consistency term by
    zero, so half of its GPU time buys nothing. Dropping the shifted view is only sound
    because CLIPVAD is batch independent -- test_two_view_batching.py proves that on the
    real model -- and because a term multiplied by zero contributes no gradient. This
    test is the end-to-end consequence: the three task losses on the first step, and the
    weights after a full epoch, must come out the same either way.
    """
    _, two_view_model, two_view_losses = _run_lambda_zero(root, list_path, "lam0_two", False)
    args, single_model, single_losses = _run_lambda_zero(root, list_path, "lam0_one", True)

    for name, a, b in zip(("loss1", "loss2", "loss3"), two_view_losses, single_losses):
        assert abs(a - b) < 1e-6, f"{name} differs between the two paths: {a} vs {b}"
    assert single_losses[3] == 0.0, "the single-view path must report a zero consistency term"

    two_view_state = two_view_model.state_dict()
    worst = max(
        (two_view_state[key] - value).abs().max().item()
        for key, value in single_model.state_dict().items()
        if value.dtype.is_floating_point and value.numel()
    )
    assert worst < 1e-5, f"weights diverged after one epoch, largest difference {worst:.3e}"
    assert Path(args.output_model_path).exists(), "final weights were not saved"
    print(f"  ok: --skip-shifted-view reproduces the two-view lambda-0 control "
          f"(largest weight difference {worst:.2e})")


def test_skip_shifted_view_refuses_a_live_lambda(root, list_path):
    """The shortcut is silent data loss if lambda is not zero, so it must refuse."""
    for kwargs in ({"lambda_consistency": 0.01}, {"lambda_auto": 0.1, "lambda_consistency": 0.0}):
        args = build_args(root, list_path, skip_shifted_view=True, **kwargs)
        model = build_model(args)
        normal_loader, anomaly_loader = build_loaders(args)
        refused = False
        try:
            train(model, normal_loader, anomaly_loader, None, args, LABEL_MAP, "cpu")
        except ValueError as error:
            refused = "--skip-shifted-view" in str(error)
        assert refused, f"expected a refusal for {kwargs}, the run went ahead instead"
    print("  ok: --skip-shifted-view refuses to run with a live consistency term")


def main():
    print("test_train_smoke")
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        list_path = build_fixture(root)
        test_checkpoint_dict_roundtrip(root)
        test_consistency_gradient_reaches_the_adapter(root, list_path)
        test_one_epoch_runs(root, list_path, branch="c", random_shift=False)
        test_one_epoch_runs(root, list_path, branch="both", random_shift=True)
        test_best_checkpoint_reload_path(root, list_path)
        test_tuned_cut_and_auto_lambda(root, list_path)
        test_lambda_recalibration_moves_and_is_capped(root, list_path)
        test_skip_shifted_view_matches_the_two_view_control(root, list_path)
        test_skip_shifted_view_refuses_a_live_lambda(root, list_path)
    print("all smoke checks passed")


if __name__ == "__main__":
    main()
