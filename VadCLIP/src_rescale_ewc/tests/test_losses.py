"""Unit tests for the rescaling and consolidation terms. CPU only, no CLIP needed.

    python -m pytest tests/test_losses.py -q          (from VadCLIP/src_rescale_ewc)
    python tests/test_losses.py                       (no pytest installed)

The first two tests are the important ones: with mu = 1 the new losses must be bit-for-bit
the upstream ones, otherwise every comparison in the experiment is against a moved
goalpost.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fisher import normalize_fisher  # noqa: E402
from losses import (  # noqa: E402
    CLAS2,
    CLASM,
    ScaleClassGrad,
    build_class_scale,
    build_video_weights,
    consolidation_penalty,
)


DEVICE = "cpu"
LABELS = ["Normal", "Explosion", "Stealing", "Shooting"]
PROMPTS = ["normal", "explosion", "stealing", "shooting"]
LABEL_MAP = {"Normal": "normal", "Explosion": "explosion",
             "Stealing": "stealing", "Shooting": "shooting"}


# --- The upstream losses, copied verbatim from ucf_train.py, as the reference ---------

def upstream_CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp).view(1)], dim=0)
    return F.binary_cross_entropy(instance_logits, labels)


def upstream_CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    return -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)


def make_batch(seed=0, batch=4, length=64, classes=4):
    generator = torch.Generator().manual_seed(seed)
    logits1 = torch.randn(batch, length, 1, generator=generator)
    logits2 = torch.randn(batch, length, classes, generator=generator)
    lengths = torch.tensor([length, length // 2, length // 4, length])
    labels = torch.zeros(batch, classes)
    for i in range(batch):
        labels[i, i % classes] = 1.0
    return logits1, logits2, labels, lengths


# --- Tests ---------------------------------------------------------------------------

def test_clas2_matches_upstream_without_weights():
    logits1, _, labels, lengths = make_batch()
    assert torch.allclose(
        CLAS2(logits1, labels, lengths, DEVICE),
        upstream_CLAS2(logits1, labels, lengths, DEVICE),
        atol=0, rtol=0,
    )


def test_clasm_matches_upstream_without_weights():
    _, logits2, labels, lengths = make_batch()
    assert torch.allclose(
        CLASM(logits2, labels, lengths, DEVICE),
        upstream_CLASM(logits2, labels, lengths, DEVICE),
        atol=0, rtol=0,
    )


def test_mu_one_is_a_no_op_in_both_modes():
    logits1, logits2, labels, lengths = make_batch()
    weights = build_video_weights(LABELS, {"Explosion"}, mu=1.0, device=DEVICE)
    scale = build_class_scale(PROMPTS, LABEL_MAP, {"Explosion"}, mu=1.0, device=DEVICE)
    assert torch.allclose(CLAS2(logits1, labels, lengths, DEVICE, weights),
                          CLAS2(logits1, labels, lengths, DEVICE))
    assert torch.allclose(CLASM(logits2, labels, lengths, DEVICE, weights, "none", scale),
                          CLASM(logits2, labels, lengths, DEVICE))


def test_video_weights_pick_out_target_classes():
    weights = build_video_weights(LABELS, {"Explosion", "Shooting"}, mu=3.0, device=DEVICE)
    assert weights.tolist() == [1.0, 3.0, 1.0, 3.0]


def test_video_rescaling_follows_equation_1():
    """L' = mean_i(w_i * l_i): a batch where only one video is a target."""
    logits1, _, labels, lengths = make_batch()
    weights = build_video_weights(LABELS, {"Explosion"}, mu=4.0, device=DEVICE)
    plain = CLAS2(logits1, labels, lengths, DEVICE)
    rescaled = CLAS2(logits1, labels, lengths, DEVICE, weights, "none")
    # Only index 1 is a target, so the increase is exactly 3 * l_1 / B.
    probabilities = torch.sigmoid(logits1).reshape(logits1.shape[0], logits1.shape[1])
    top, _ = torch.topk(probabilities[1, 0:lengths[1]], k=int(lengths[1] / 16 + 1), largest=True)
    per_video_1 = F.binary_cross_entropy(top.mean().view(1), (1 - labels[:, 0])[1:2])
    assert torch.allclose(rescaled - plain, 3.0 * per_video_1 / logits1.shape[0], atol=1e-6)


def test_normalize_mean_keeps_the_loss_scale():
    logits1, _, labels, lengths = make_batch()
    weights = build_video_weights(LABELS, {"Explosion", "Shooting"}, mu=5.0, device=DEVICE)
    assert (CLAS2(logits1, labels, lengths, DEVICE, weights, "mean")
            < CLAS2(logits1, labels, lengths, DEVICE, weights, "none"))


def test_class_scale_leaves_the_loss_value_untouched():
    """Eq. (10)-(11) rescale gradients only; the reported loss must not move."""
    _, logits2, labels, lengths = make_batch()
    scale = build_class_scale(PROMPTS, LABEL_MAP, {"Explosion"}, mu=7.0, device=DEVICE)
    assert torch.allclose(CLASM(logits2, labels, lengths, DEVICE, class_scale=scale),
                          CLASM(logits2, labels, lengths, DEVICE), atol=1e-7)


def test_class_scale_multiplies_only_the_target_columns_gradient():
    _, logits2, labels, lengths = make_batch()
    mu = 6.0
    target_index = PROMPTS.index("explosion")

    plain_input = logits2.clone().requires_grad_(True)
    CLASM(plain_input, labels, lengths, DEVICE).backward()
    plain_grad = plain_input.grad.clone()

    scaled_input = logits2.clone().requires_grad_(True)
    scale = build_class_scale(PROMPTS, LABEL_MAP, {"Explosion"}, mu=mu, device=DEVICE)
    CLASM(scaled_input, labels, lengths, DEVICE, class_scale=scale).backward()
    scaled_grad = scaled_input.grad.clone()

    other = [j for j in range(logits2.shape[-1]) if j != target_index]
    assert torch.allclose(scaled_grad[..., target_index], mu * plain_grad[..., target_index], atol=1e-6)
    assert torch.allclose(scaled_grad[..., other], plain_grad[..., other], atol=1e-6)


def test_class_scale_reaches_normal_videos_too():
    """The push-down gradient on non-target videos is scaled as well.

    This is the counterpart of the paper's blank-token finding (section 5.4): emphasising
    only the push-up direction drives the model to a degenerate solution.
    """
    _, logits2, labels, lengths = make_batch()
    target_index = PROMPTS.index("explosion")
    normal_video = 0  # its label is 'normal', not the target class

    plain_input = logits2.clone().requires_grad_(True)
    CLASM(plain_input, labels, lengths, DEVICE).backward()
    reference = plain_input.grad[normal_video, :, target_index].clone()

    scaled_input = logits2.clone().requires_grad_(True)
    scale = build_class_scale(PROMPTS, LABEL_MAP, {"Explosion"}, mu=3.0, device=DEVICE)
    CLASM(scaled_input, labels, lengths, DEVICE, class_scale=scale).backward()

    assert reference.abs().sum() > 0
    assert torch.allclose(scaled_input.grad[normal_video, :, target_index], 3.0 * reference, atol=1e-6)


def test_scale_class_grad_is_identity_forward():
    x = torch.randn(3, 5, requires_grad=True)
    scale = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    assert torch.equal(ScaleClassGrad.apply(x, scale), x)


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(3, 2, bias=False)
        self.frozen = torch.nn.Linear(3, 2, bias=False)
        self.frozen.weight.requires_grad = False


def test_consolidation_is_zero_at_the_anchor():
    model = _TinyModel()
    anchor = {"a.weight": model.a.weight.detach().clone()}
    fisher = {"a.weight": torch.rand_like(model.a.weight)}
    assert float(consolidation_penalty(model, anchor, fisher, lam=1000.0).detach()) == 0.0


def test_consolidation_matches_equations_12_and_13():
    model = _TinyModel()
    anchor = {"a.weight": torch.zeros_like(model.a.weight)}
    fisher = {"a.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    lam = 2.0

    l2 = consolidation_penalty(model, anchor, None, lam)           # Eq. (12)
    ewc = consolidation_penalty(model, anchor, fisher, lam)        # Eq. (13)
    squared = model.a.weight.detach().pow(2)
    assert torch.allclose(l2, 0.5 * lam * squared.sum(), atol=1e-6)
    assert torch.allclose(ewc, 0.5 * lam * (fisher["a.weight"] * squared).sum(), atol=1e-6)


def test_consolidation_skips_frozen_parameters():
    """VadCLIP freezes the CLIP backbone; those weights must never enter the penalty."""
    model = _TinyModel()
    anchor = {"a.weight": torch.zeros_like(model.a.weight),
              "frozen.weight": torch.zeros_like(model.frozen.weight)}
    penalty = consolidation_penalty(model, anchor, None, lam=2.0)
    expected = 0.5 * 2.0 * model.a.weight.detach().pow(2).sum()
    assert torch.allclose(penalty, expected, atol=1e-6)


def test_normalize_fisher_gives_unit_mean():
    fisher = {"a": torch.rand(100) * 7, "b": torch.rand(50) * 0.3}
    normalized = normalize_fisher(fisher, "mean")
    flat = torch.cat([value.reshape(-1) for value in normalized.values()])
    assert abs(float(flat.mean()) - 1.0) < 1e-5


def test_quantiles_follow_the_input_device_and_dtype():
    """torch.quantile rejects a q tensor whose device or dtype differs from the input.

    The device half of that only shows up on CUDA, which this suite cannot reach; the
    dtype half is the same bug and is checkable here.
    """
    from fisher import _quantiles

    for dtype in (torch.float32, torch.float64):
        values = _quantiles(torch.rand(1000, dtype=dtype))
        assert len(values) == 4
        assert values == sorted(values)


def test_fisher_statistics_reports_the_expected_keys():
    from fisher import fisher_statistics

    statistics = fisher_statistics({"a": torch.rand(500), "b": torch.zeros(10)})
    for key in ("num_parameters", "mean", "max", "min", "zero_fraction",
                "median", "p90", "p99", "p999"):
        assert key in statistics, key
    assert statistics["num_parameters"] == 510
    assert statistics["min"] == 0.0


def test_normalize_fisher_max_bounds_at_one():
    fisher = {"a": torch.rand(100) * 7, "b": torch.rand(50) * 0.3}
    flat = torch.cat([v.reshape(-1) for v in normalize_fisher(fisher, "max").values()])
    assert abs(float(flat.max()) - 1.0) < 1e-6


def test_fisher_squares_per_sample_not_per_batch():
    """Eq. (14) squares each sample's gradient. Cancelling gradients must not vanish."""
    from fisher import accumulate_squared_gradients, finalize_fisher, zero_fisher

    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    fisher = zero_fisher(model)
    # Two samples with exactly opposite gradients: the mean gradient is 0, but F is not.
    for sign in (1.0, -1.0):
        model.zero_grad(set_to_none=True)
        accumulate_squared_gradients(model, (model.weight * sign).sum(), fisher)
    fisher = finalize_fisher(fisher, 2)
    assert torch.allclose(fisher["weight"], torch.ones_like(fisher["weight"]))


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"PASS  {name}")
            except Exception as error:  # noqa: BLE001
                failures += 1
                print(f"FAIL  {name}: {type(error).__name__}: {error}")
    print(f"\n{'ALL PASSED' if failures == 0 else str(failures) + ' FAILED'}")
    sys.exit(1 if failures else 0)
