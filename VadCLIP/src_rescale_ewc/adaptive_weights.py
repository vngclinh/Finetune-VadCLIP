"""Per-class gradient weights for VadCLIP stage 2, derived from the train set alone.

``build_class_scale`` gives every hand-picked target class the same ``mu``. Two things are
wrong with that. The target set was picked because those classes score worst *on the test
set*, so the test set leaks into the design of the method; and one shared ``mu`` says a
class is either emphasised or not, with nothing in between.

This module replaces both with a weight vector computed from two train-set signals:

    frequency   how many training videos the class has. Rarer -> fewer gradient updates.
    difficulty  how badly the source checkpoint already handles the class on train.

Both are turned into a [0, 1] score by *rank*, then mixed and stretched:

    freq_c = (median_count / count_c) ** p      -> rank_normalize -> [0, 1]
    diff_c = per-class loss from the source     -> rank_normalize -> [0, 1]
    mix_c  = beta * diff_c + (1 - beta) * freq_c
    w_c    = clip(1 + alpha * mix_c, 1, w_max)

Three deliberate choices, each of which fixes a way the obvious formula misbehaves on this
dataset:

* **Rank, not min-max.** Min-max normalisation is set entirely by the two extreme classes,
  so a single outlier (``Abuse`` has 48 train videos but only 2 test videos) rescales
  everything. Ranks are immune to that.
* **Additive mix, not ``diff * freq``.** With a product, dropping one factor does not give
  you "the other signal alone", it gives you a constant times the other signal: at
  ``beta = 0`` the multiplicative form hands *every* class, ``Normal`` included, a weight
  of at least ``1 + alpha * min(freq)``. That is a global learning-rate change on the A
  branch, not a frequency ablation. Additively, ``beta`` interpolates between two signals
  that already share the [0, 1] range, so beta = 0 / 0.5 / 1 is a real ablation.
* **``Normal`` is pinned to 1.0.** It is not an anomaly class waiting to be boosted, and
  with 800 videos against a median of 45 the frequency term would peg it at the floor
  anyway. Emphasising the normal class is a different experiment; it should not happen by
  accident through a formula.

``w_max`` is a guard rail, not a knob: at the defaults (alpha 6) the weights span 1 to 7
and never reach it. It exists so a larger alpha cannot silently produce the 12x gradients
that made the from-scratch runs diverge.

Pure stdlib on purpose -- no torch, no numpy -- so the formula is unit-testable on its own
and ``ucf_train_difficulty.py`` is the only thing that needs a GPU.
"""

import json
import math
from datetime import datetime, timezone

DEFAULT_ALPHA = 6.0
DEFAULT_W_MAX = 8.0
DEFAULT_BETA = 0.5
DEFAULT_FREQUENCY_POWER = 0.5

# Excluded from the ranking and pinned at 1.0. See the module docstring.
PINNED_CLASSES = ("Normal",)


def _average_ranks(values):
    """0-based ranks; tied values all take the mean of the ranks they span."""
    ranks = {}
    order = sorted(values)
    index = 0
    while index < len(order):
        last = index
        while last + 1 < len(order) and order[last + 1] == order[index]:
            last += 1
        ranks[order[index]] = (index + last) / 2.0
        index = last + 1
    return ranks


def rank_normalize(mapping):
    """Map values onto [0, 1] by rank: smallest -> 0, largest -> 1.

    An all-equal input carries no ordering information, so it returns 0 everywhere rather
    than 0.5 everywhere. That matters: 0.5 everywhere would amplify every class equally,
    which is a learning-rate change wearing a weight vector's clothes.
    """
    if not mapping:
        return {}
    values = list(mapping.values())
    if max(values) == min(values):
        return {key: 0.0 for key in mapping}
    ranks = _average_ranks(values)
    span = len(mapping) - 1
    return {key: ranks[value] / span for key, value in mapping.items()}


def frequency_signal(counts, power=DEFAULT_FREQUENCY_POWER):
    """``(median_count / count_c) ** power`` -- the raw, un-normalised rarity of a class.

    The median is taken over the classes passed in, so pinned classes must already be out;
    ``Normal``'s 800 videos would drag the median far off the anomaly classes' scale.
    """
    if not counts:
        return {}
    ordered = sorted(counts.values())
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
    return {name: (median / count) ** power if count > 0 else 0.0
            for name, count in counts.items()}


def spearman(first, second):
    """Rank correlation of two dicts over their shared keys. None if it is undefined.

    This is the number that decides whether "frequency" and "difficulty" are two signals
    or one. The source checkpoint was trained on this very train set, so a class with few
    videos got few gradient updates and keeps a high train loss -- difficulty may be little
    more than frequency in disguise. A high |rho| here means the ablation between them is
    not measuring what it claims to.
    """
    keys = [key for key in first if key in second]
    if len(keys) < 3:
        return None
    first_ranks = _average_ranks([first[key] for key in keys])
    second_ranks = _average_ranks([second[key] for key in keys])
    xs = [first_ranks[first[key]] for key in keys]
    ys = [second_ranks[second[key]] for key in keys]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    spread_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    spread_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if spread_x == 0 or spread_y == 0:
        return None
    return covariance / (spread_x * spread_y)


def compute_adaptive_weights(counts, difficulty, alpha=DEFAULT_ALPHA, w_max=DEFAULT_W_MAX,
                             beta=DEFAULT_BETA, frequency_power=DEFAULT_FREQUENCY_POWER,
                             pinned=PINNED_CLASSES):
    """Return ``{class: weight}`` plus the per-class breakdown that produced it.

    ``beta`` selects the ablation arm: 1.0 difficulty only, 0.0 frequency only, 0.5 both.
    Classes in ``pinned`` are excluded from every ranking and get exactly 1.0.
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"--adaptive-beta must be in [0, 1]. Got {beta}.")
    if alpha < 0:
        raise ValueError(f"--adaptive-alpha must be >= 0. Got {alpha}.")
    if w_max < 1.0:
        raise ValueError(f"--adaptive-w-max must be >= 1. Got {w_max}.")

    pinned = set(pinned)
    ranked = [name for name in difficulty if name not in pinned]
    missing = [name for name in ranked if name not in counts]
    if missing:
        raise KeyError(f"No train video count for {sorted(missing)}.")

    raw_frequency = frequency_signal({name: counts[name] for name in ranked}, frequency_power)
    normalized_frequency = rank_normalize(raw_frequency)
    normalized_difficulty = rank_normalize({name: difficulty[name] for name in ranked})

    weights, breakdown = {}, {}
    for name in ranked:
        mix = beta * normalized_difficulty[name] + (1.0 - beta) * normalized_frequency[name]
        weight = min(max(1.0 + alpha * mix, 1.0), w_max)
        weights[name] = weight
        breakdown[name] = {
            "train_video_count": counts[name],
            "difficulty_raw": difficulty[name],
            "frequency_raw": raw_frequency[name],
            "difficulty_score": normalized_difficulty[name],
            "frequency_score": normalized_frequency[name],
            "mix": mix,
            "adaptive_weight": weight,
        }
    for name in pinned:
        weights[name] = 1.0
        breakdown[name] = {
            "train_video_count": counts.get(name, 0),
            "difficulty_raw": difficulty.get(name),
            "frequency_raw": None,
            "difficulty_score": None,
            "frequency_score": None,
            "mix": None,
            "adaptive_weight": 1.0,
            "pinned": True,
        }
    return weights, breakdown


def build_weight_document(weights, breakdown, hyperparameters, extra=None):
    """The on-disk format. ``weights`` is the only part the trainer reads."""
    document = {
        "schema": "vadclip-adaptive-class-weights/1",
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hyperparameters": dict(hyperparameters),
        "weights": {name: round(value, 6) for name, value in sorted(weights.items())},
        "breakdown": breakdown,
    }
    if extra:
        document.update(extra)
    return document


def save_weight_document(path, document):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=False)
        handle.write("\n")


def load_class_weights(path):
    """Read ``{class: weight}`` back, with the errors a typo actually produces.

    A stage-2 run that silently fell back to all-ones would look like a control run and
    quietly poison the comparison table, so every failure here is loud.
    """
    with open(path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    if "weights" not in document:
        raise KeyError(f"{path} has no 'weights' object. Was it written by "
                       f"ucf_train_difficulty.py?")
    weights = document["weights"]
    if not weights:
        raise ValueError(f"{path} carries an empty weight vector.")
    for name, value in weights.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{path}: weight for {name!r} is {value!r}, not a number.")
        if value <= 0:
            raise ValueError(f"{path}: weight for {name!r} is {value}, which must be > 0.")
    hyperparameters = document.get("hyperparameters", {})
    return {name: float(value) for name, value in weights.items()}, hyperparameters


def format_weight_table(breakdown, correlation=None):
    """The table to paste into the report: one line per class, heaviest first."""
    header = (f"{'class':<16}{'videos':>7}{'diff_raw':>10}{'freq':>7}{'diff':>7}"
              f"{'mix':>7}{'weight':>8}")
    lines = [header, "-" * len(header)]

    def cell(row, key, width, digits):
        value = row.get(key)
        if value is None:
            return f"{'-':>{width}}"
        return f"{value:>{width}.{digits}f}"

    for name, row in sorted(breakdown.items(), key=lambda item: -item[1]["adaptive_weight"]):
        lines.append(
            f"{name:<16}{row['train_video_count']:>7}"
            f"{cell(row, 'difficulty_raw', 10, 4)}{cell(row, 'frequency_score', 7, 2)}"
            f"{cell(row, 'difficulty_score', 7, 2)}{cell(row, 'mix', 7, 2)}"
            f"{row['adaptive_weight']:>8.3f}"
        )
    if correlation is not None:
        lines.append("")
        lines.append(f"Spearman(train_video_count, difficulty_raw) = {correlation:+.3f}")
        lines.append(correlation_verdict(correlation))
    return "\n".join(lines)


def correlation_verdict(correlation):
    """Say out loud what the correlation means for the beta ablation."""
    magnitude = abs(correlation)
    if magnitude >= 0.6:
        return ("  -> STRONG. Difficulty is largely frequency restated: the source model was "
                "trained on\n     this very train set, so rare classes keep a high train loss. "
                "Report this, and treat\n     beta=0 vs beta=1 as two views of one signal, not "
                "as an ablation of two.")
    if magnitude >= 0.3:
        return ("  -> MODERATE. The two signals overlap but are not identical. The beta ablation "
                "is\n     interpretable; say in the report that they are correlated.")
    return ("  -> WEAK. Difficulty carries information frequency does not. The beta ablation is\n"
            "     measuring two genuinely different things.")
