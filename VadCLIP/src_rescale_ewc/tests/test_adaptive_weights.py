"""Unit tests for the adaptive per-class weights. CPU only, no CLIP needed.

    python -m pytest tests/test_adaptive_weights.py -q     (from VadCLIP/src_rescale_ewc)
    python tests/test_adaptive_weights.py                  (no pytest installed)

Three of these are load-bearing for the experiment rather than for the code:

* ``test_beta_zero_leaves_the_easiest_class_unamplified`` pins down the reason the mix is
  additive. The multiplicative form the plan started from gives every class, Normal
  included, a weight of at least ``1 + alpha * min(freq)`` once difficulty is dropped --
  a global A-branch learning-rate change masquerading as a frequency ablation.
* ``test_difficulty_and_frequency_can_disagree`` is the case the whole method exists for:
  a class with many videos that the source model still handles badly (RoadAccidents has
  127 train videos and is one of the four weak classes). Frequency alone cannot see it.
* ``test_gradient_is_scaled_column_by_column`` is a regression test on machinery that
  already worked -- ``ScaleClassGrad`` took a vector before this change and only ever got
  a constant written into it -- so it stays correct once real per-class values arrive.
"""

import json
import math
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptive_weights import (  # noqa: E402
    PINNED_CLASSES,
    build_weight_document,
    compute_adaptive_weights,
    frequency_signal,
    load_class_weights,
    rank_normalize,
    save_weight_document,
    spearman,
)
from losses import (  # noqa: E402
    CLASM,
    ScaleClassGrad,
    build_class_scale_from_weights,
    clas2_per_video,
    clasm_per_video,
)

DEVICE = "cpu"
LABEL_MAP = {"Normal": "normal", "Explosion": "explosion",
             "Stealing": "stealing", "Shooting": "shooting"}
PROMPTS = ["normal", "explosion", "stealing", "shooting"]

# The real UCF-Crime train distribution, so the tests exercise the shape of data the
# experiment actually meets rather than a tidy invented one.
UCF_COUNTS = {
    "Normal": 800, "Robbery": 145, "RoadAccidents": 127, "Stealing": 95, "Burglary": 87,
    "Abuse": 48, "Assault": 47, "Vandalism": 45, "Fighting": 45, "Arrest": 45,
    "Arson": 41, "Shoplifting": 29, "Explosion": 29, "Shooting": 27,
}


# --- rank_normalize ------------------------------------------------------------------

def test_rank_normalize_spans_zero_to_one():
    result = rank_normalize({"a": 5.0, "b": 1.0, "c": 3.0})
    assert result == {"b": 0.0, "c": 0.5, "a": 1.0}


def test_rank_normalize_is_immune_to_an_outlier():
    """The point of ranks over min-max: one extreme value must not rescale the others."""
    tame = rank_normalize({"a": 1.0, "b": 2.0, "c": 3.0})
    with_outlier = rank_normalize({"a": 1.0, "b": 2.0, "c": 1000.0})
    assert tame == with_outlier


def test_rank_normalize_of_a_flat_input_is_zero_not_a_half():
    """No ordering means no signal. 0.5 everywhere would amplify every class equally."""
    assert rank_normalize({"a": 2.0, "b": 2.0, "c": 2.0}) == {"a": 0.0, "b": 0.0, "c": 0.0}


def test_rank_normalize_averages_ties():
    result = rank_normalize({"a": 1.0, "b": 2.0, "c": 2.0, "d": 3.0})
    assert result["a"] == 0.0 and result["d"] == 1.0
    assert result["b"] == result["c"] == 0.5  # ranks 1 and 2 share their mean, 1.5 / 3


# --- frequency_signal ----------------------------------------------------------------

def test_frequency_signal_rewards_rarity():
    signal = frequency_signal({"rare": 25, "median": 100, "common": 400}, power=0.5)
    assert signal["rare"] > signal["median"] > signal["common"]
    assert math.isclose(signal["median"], 1.0)


def test_frequency_power_does_not_change_the_ordering():
    counts = {name: value for name, value in UCF_COUNTS.items() if name != "Normal"}
    half = rank_normalize(frequency_signal(counts, 0.5))
    whole = rank_normalize(frequency_signal(counts, 1.0))
    assert half == whole


# --- compute_adaptive_weights --------------------------------------------------------

def make_difficulty(counts, **overrides):
    """A difficulty that is a pure function of rarity, unless a class is overridden."""
    difficulty = {name: 1.0 / count for name, count in counts.items() if name != "Normal"}
    difficulty.update(overrides)
    return difficulty


def test_normal_is_pinned_to_one():
    difficulty = make_difficulty(UCF_COUNTS)
    difficulty["Normal"] = 99.0  # even claiming it is the hardest class must not matter
    weights, breakdown = compute_adaptive_weights(UCF_COUNTS, difficulty)
    assert weights["Normal"] == 1.0
    assert breakdown["Normal"]["pinned"] is True


def test_weights_stay_inside_one_to_one_plus_alpha():
    weights, _ = compute_adaptive_weights(UCF_COUNTS, make_difficulty(UCF_COUNTS), alpha=6.0)
    anomaly = [value for name, value in weights.items() if name not in PINNED_CLASSES]
    assert min(anomaly) == 1.0
    assert max(anomaly) == 7.0


def test_w_max_clips():
    weights, _ = compute_adaptive_weights(UCF_COUNTS, make_difficulty(UCF_COUNTS),
                                          alpha=20.0, w_max=8.0)
    assert max(weights.values()) == 8.0


def test_beta_zero_leaves_the_easiest_class_unamplified():
    """The additive mix's reason to exist: beta=0 must be an ablation, not a global boost."""
    weights, _ = compute_adaptive_weights(UCF_COUNTS, make_difficulty(UCF_COUNTS), beta=0.0)
    anomaly = {name: value for name, value in weights.items() if name not in PINNED_CLASSES}
    assert min(anomaly.values()) == 1.0
    # Robbery is the most common anomaly class, so frequency alone must not touch it.
    assert anomaly["Robbery"] == 1.0
    assert anomaly["Shooting"] == 7.0  # the rarest one takes the full stretch


def test_difficulty_and_frequency_can_disagree():
    """A common-but-hard class is invisible to frequency and obvious to difficulty."""
    difficulty = make_difficulty(UCF_COUNTS, RoadAccidents=10.0)  # 127 videos, still hard

    by_frequency, _ = compute_adaptive_weights(UCF_COUNTS, difficulty, beta=0.0)
    by_difficulty, _ = compute_adaptive_weights(UCF_COUNTS, difficulty, beta=1.0)

    assert by_frequency["RoadAccidents"] < 2.0   # near the floor: it is not rare
    assert by_difficulty["RoadAccidents"] == 7.0  # at the ceiling: it is the hardest

    both, _ = compute_adaptive_weights(UCF_COUNTS, difficulty, beta=0.5)
    assert by_frequency["RoadAccidents"] < both["RoadAccidents"] < by_difficulty["RoadAccidents"]


def test_harder_class_never_gets_a_smaller_weight():
    difficulty = make_difficulty(UCF_COUNTS, Explosion=5.0, Arson=0.001)
    weights, _ = compute_adaptive_weights(UCF_COUNTS, difficulty, beta=1.0)
    assert weights["Explosion"] > weights["Arson"]


def test_missing_count_is_an_error_not_a_default():
    difficulty = make_difficulty(UCF_COUNTS)
    difficulty["Loitering"] = 1.0
    try:
        compute_adaptive_weights(UCF_COUNTS, difficulty)
    except KeyError as error:
        assert "Loitering" in str(error)
    else:
        raise AssertionError("a class with no train count must not silently get a weight")


def test_bad_hyperparameters_are_rejected():
    difficulty = make_difficulty(UCF_COUNTS)
    for kwargs in ({"beta": 1.5}, {"beta": -0.1}, {"alpha": -1.0}, {"w_max": 0.5}):
        try:
            compute_adaptive_weights(UCF_COUNTS, difficulty, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"{kwargs} should have been rejected")


# --- spearman ------------------------------------------------------------------------

def test_spearman_endpoints():
    rising = {"a": 1, "b": 2, "c": 3, "d": 4}
    assert math.isclose(spearman(rising, {"a": 10, "b": 20, "c": 30, "d": 40}), 1.0)
    assert math.isclose(spearman(rising, {"a": 40, "b": 30, "c": 20, "d": 10}), -1.0)
    assert spearman(rising, {"a": 1, "b": 1, "c": 1, "d": 1}) is None


def test_spearman_is_monotone_not_linear():
    rising = {"a": 1, "b": 2, "c": 3, "d": 4}
    assert math.isclose(spearman(rising, {"a": 1, "b": 8, "c": 27, "d": 64}), 1.0)


# --- the weight file -----------------------------------------------------------------

def test_weight_file_round_trip():
    weights, breakdown = compute_adaptive_weights(UCF_COUNTS, make_difficulty(UCF_COUNTS))
    document = build_weight_document(weights, breakdown, {"alpha": 6.0, "beta": 0.5})
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "adaptive_weights.json"
        save_weight_document(path, document)
        loaded, meta = load_class_weights(path)
        assert meta["beta"] == 0.5
        for name, value in weights.items():
            assert math.isclose(loaded[name], value, abs_tol=1e-6)


def test_broken_weight_files_are_rejected_loudly():
    """A run that silently fell back to all-ones would look like a control run."""
    bad_documents = [
        ({"hyperparameters": {}}, KeyError),           # no weights at all
        ({"weights": {}}, ValueError),                 # empty vector
        ({"weights": {"Normal": "1.0"}}, TypeError),   # a string that looks like a number
        ({"weights": {"Normal": True}}, TypeError),    # bool is an int in Python; not here
        ({"weights": {"Normal": 0.0}}, ValueError),    # zero kills the class' gradient
        ({"weights": {"Normal": -2.0}}, ValueError),   # negative reverses it
    ]
    with tempfile.TemporaryDirectory() as directory:
        for index, (document, expected) in enumerate(bad_documents):
            path = Path(directory) / f"bad_{index}.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            try:
                load_class_weights(path)
            except expected:
                continue
            raise AssertionError(f"{document} should have raised {expected.__name__}")


# --- build_class_scale_from_weights ---------------------------------------------------

def test_scale_vector_lands_on_the_right_columns():
    weights = {"Normal": 1.0, "Explosion": 6.5, "Stealing": 2.0, "Shooting": 3.25}
    scale = build_class_scale_from_weights(PROMPTS, LABEL_MAP, weights, DEVICE)
    assert torch.allclose(scale, torch.tensor([1.0, 6.5, 2.0, 3.25]))


def test_partial_weight_file_is_an_error():
    """Silence here would run an experiment nobody described: some classes at 1.0 by accident."""
    try:
        build_class_scale_from_weights(PROMPTS, LABEL_MAP, {"Explosion": 6.5}, DEVICE)
    except KeyError as error:
        assert "Shooting" in str(error)
    else:
        raise AssertionError("a weight file missing classes must be rejected")


def test_unknown_class_in_the_weight_file_is_an_error():
    weights = {name: 1.0 for name in LABEL_MAP}
    weights["Loitering"] = 4.0
    try:
        build_class_scale_from_weights(PROMPTS, LABEL_MAP, weights, DEVICE)
    except KeyError as error:
        assert "Loitering" in str(error)
    else:
        raise AssertionError("a weight file from a different class set must be rejected")


# --- the gradient itself --------------------------------------------------------------

def make_batch(seed=0, batch=6, length=64, classes=4):
    generator = torch.Generator().manual_seed(seed)
    logits1 = torch.randn(batch, length, 1, generator=generator)
    logits2 = torch.randn(batch, length, classes, generator=generator)
    labels = torch.zeros(batch, classes)
    for row in range(batch):
        labels[row, row % classes] = 1.0
    lengths = torch.full((batch,), length)
    return logits1, logits2, labels, lengths


def test_scale_class_grad_multiplies_each_column():
    scale = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x = torch.randn(3, 4, requires_grad=True)
    ScaleClassGrad.apply(x, scale).sum().backward()
    assert torch.allclose(x.grad, scale.expand(3, 4))


def test_gradient_is_scaled_column_by_column():
    """Per-class weights must reach the A-branch logits class by class, exactly.

    The forward pass is the identity, so the same frames win the top-k either way. The
    gradient on class column c must therefore come out exactly ``scale[c]`` times the
    unscaled one -- not merely larger.
    """
    _, logits2, labels, lengths = make_batch()
    scale = torch.tensor([1.0, 6.5, 2.0, 3.25])

    plain = logits2.clone().requires_grad_(True)
    clasm_per_video(plain, labels, lengths, DEVICE)[0].sum().backward()

    scaled = logits2.clone().requires_grad_(True)
    clasm_per_video(scaled, labels, lengths, DEVICE, scale)[0].sum().backward()

    for column in range(scale.numel()):
        assert torch.allclose(scaled.grad[:, :, column],
                              plain.grad[:, :, column] * scale[column], atol=1e-6)


def test_all_ones_weights_are_a_no_op():
    """The adaptive control arm must be bit-for-bit the unrescaled loss."""
    _, logits2, labels, lengths = make_batch(seed=3)
    ones = build_class_scale_from_weights(PROMPTS, LABEL_MAP,
                                          {name: 1.0 for name in LABEL_MAP}, DEVICE)
    with_scale = CLASM(logits2, labels, lengths, DEVICE, class_scale=ones)
    without = CLASM(logits2, labels, lengths, DEVICE)
    assert torch.equal(with_scale, without)


def test_per_video_helpers_reduce_to_the_batch_losses():
    """The refactor that let the difficulty script reuse the trainer's numbers."""
    logits1, logits2, labels, lengths = make_batch(seed=7)
    per_video, mil_score, targets = clas2_per_video(logits1, labels, lengths, DEVICE)
    assert per_video.shape == (logits1.shape[0],)
    assert mil_score.shape == targets.shape == per_video.shape

    from losses import CLAS2
    assert torch.allclose(per_video.mean(), CLAS2(logits1, labels, lengths, DEVICE))
    assert torch.allclose(clasm_per_video(logits2, labels, lengths, DEVICE)[0].mean(),
                          CLASM(logits2, labels, lengths, DEVICE))


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
