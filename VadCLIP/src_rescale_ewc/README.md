# Loss rescaling + weight consolidation for VadCLIP

Stage-2 fine-tuning of VadCLIP on UCF-Crime that applies the two ideas of
**arXiv:2302.09723** ("Emphasizing Unseen Words", Qu et al. 2023) to the
under-performing anomaly classes:

* **Loss rescaling** — stop the objective from treating every video and every class
  equally (paper Eq. 1 and Eq. 10–11).
* **Weight consolidation** — keep the emphasis from destroying what stage 1 already
  learned, weighing each parameter by its diagonal Fisher information (paper Eq. 12–14).

Background reading in `docs/`:
[paper analysis](../../docs/Paper_2302.09723_LossRescaling_EWC_Analysis.md) ·
[application plan](../../docs/Plan_LossRescaling_EWC_for_VadCLIP.md)

Nothing in `VadCLIP/src` or `VadCLIP/baseline` is modified. `model.py`, `clip/` and
`utils/` are imported from `VadCLIP/src` through `_bootstrap.py`, so the architecture and
the feature processing are byte-for-byte the ones the baseline uses.

## Files

| File | Role |
|---|---|
| `losses.py` | Weighted `CLAS2`/`CLASM`, `ScaleClassGrad` (Eq. 10–11), `consolidation_penalty` (Eq. 12/13). No CLIP import |
| `adaptive_weights.py` | The per-class weight formula: rank-normalised frequency and difficulty, mixed by `beta`. Pure stdlib |
| `ucf_train_difficulty.py` | Measure per-class difficulty on the **train** set and write the weight vector |
| `fisher.py` | Estimate / normalise / save / load the diagonal Fisher (Eq. 14). No CLIP import |
| `dataset_rescale.py` | UCF feature dataset with `--feature-root` support and a `labels` property |
| `evaluation.py` | One scoring pass → overall metrics + per-class AUC/AP + target vs non-target split |
| `ucf_option_rescale.py` | All command-line options |
| `ucf_fisher.py` | Run once on the stage-1 checkpoint to produce `fisher_ucf.pt` |
| `ucf_train_rescale.py` | The stage-2 trainer |
| `ucf_eval_perclass.py` | Score several checkpoints and print the trade-off table |
| `tests/test_losses.py` | 18 unit tests, CPU only, no CLIP needed |
| `tests/test_adaptive_weights.py` | 25 unit tests for the weight formula and the vector gradient path |

## The objective

```
L  =  L1'  +  L2'  +  L3  +  (lambda/2) * sum_i F_i (theta_i - theta'_i)^2
```

`L3` is never rescaled — it is a property of the 14 text vectors and has no video
dimension. `theta'` is the converged stage-1 model.

### `--rescale-mode video` — Eq. (1)

```
L1' = (1/B) sum_i w_i * BCE(s_i, y_i)          w_i = mu if class(i) in T else 1
L2' = (1/B) sum_i w_i * (-log softmax(z_i)[c_i])
```

Dividing by `B` rather than `sum(w)` is what the paper does, so the gradient magnitude
grows with `mu`. That is why the defaults also drop the learning rate and clip gradients.
`--rescale-normalize mean` divides by `sum(w)` instead if you want the scale held fixed.

### `--rescale-mode class` — Eq. (10)–(11)

```
d L2 / d z[i, c]  <-  mu * d L2 / d z[i, c]     for every c in T, for every video i
```

The **loss value does not change**; only the gradient does. Two consequences worth
knowing before reading the logs:

* `l2` in the training log is directly comparable to a run with `mu = 1`.
* The scaling deliberately reaches non-target videos too, where the gradient at a target
  column is the push-*down* term. Emphasising only the push-up direction collapses
  precision — the counterpart of the paper's blank-token finding (section 5.4).

The C branch is left untouched in this mode. It has a single output channel, so "the
target class' column" does not exist there; and the paper froze its attention decoder for
the same reason — a rescaled branch cannot be balanced against an untouched one.

### `--rescale-mode adaptive_class` — the same mechanism, a weight per class

Same gradient surgery as `class`, but the multiplier is `w[c]` from a file rather than one
shared `mu` over a hand-picked target set:

```
d L2 / d z[i, c]  <-  w[c] * d L2 / d z[i, c]      for every class c, for every video i
```

Two problems with the fixed form this fixes.

**The target set was chosen on the test set.** `Explosion`, `RoadAccidents`, `Shooting`
and `Shoplifting` are the four lowest-AUC classes *on test*, so the test set sits inside
the design of the method. `w` is computed from the train set alone:

```
freq_c = (median_count / count_c) ** p    -> rank-normalise -> [0, 1]
diff_c = source model's per-class loss    -> rank-normalise -> [0, 1]     (on train)
w_c    = clip(1 + alpha * (beta * diff_c + (1 - beta) * freq_c), 1, w_max)
```

**One `mu` is a switch, not a dial.** `RoadAccidents` has 127 train videos and is still
one of the weak classes; frequency alone cannot see that, and a shared `mu` cannot say
"emphasise this one twice as much as that one".

`beta` is the ablation axis: `1.0` difficulty only, `0.0` frequency only, `0.5` both.
Three details that are load-bearing rather than cosmetic:

* The mix is **additive**, over two signals already rank-normalised to `[0, 1]`. The
  multiplicative form `1 + alpha * diff * freq` looks equivalent but is not: at
  `beta = 0` it hands every class — `Normal` included — a weight of at least
  `1 + alpha * min(freq)`, which is a global A-branch learning-rate change, not a
  frequency ablation.
* Ranks, **not** min-max. Min-max is set entirely by the two extreme classes, so one
  outlier rescales everything.
* `Normal` is **pinned to 1.0** and excluded from both rankings. Emphasising the normal
  class is a different experiment and should not happen by accident through a formula.

`--target-classes` still exists in this mode, but only to decide which classes the report
aggregates as "target". It no longer steers anything the model sees — the trainer says so
in its own log line.

`--mu` is ignored here.

## Running it

### 1. Fisher (once, on the stage-1 checkpoint)

```bash
python ucf_fisher.py \
  --pretrained-model-path model/model_ucf.pth \
  --feature-root /path/to/UCFClipFeatures \
  --train-list ../list/ucf_CLIP_rgb_relative.csv \
  --fisher-path model/fisher_ucf.pt \
  --fisher-max-samples 1000
```

Prints the distribution of `F` and the mean `F` per module. Expect a very heavy tail —
after mean-normalisation the maximum sits four orders of magnitude above the median. That
gap *is* the difference between EWC and plain L2; if it were flat, Eq. (13) would collapse
into Eq. (12).

`theta'` is stored in the same file, so `--regularizer l2` needs it too.

### 2. Pick lambda

Do not copy the paper's `1e7`; it encodes the gradient scale of CTC on LibriSpeech. Either

```bash
--lambda-auto 0.15 --lambda-auto-steps 50     # solve for it
```

which runs 50 unconstrained steps, measures `Q = sum_i F_i (theta_i - theta'_i)^2` and
sets `lambda = 2 * 0.15 * L_task / Q`, or read the `reg/task` column the trainer prints
and set `--lambda-reg` by hand. Aim for roughly **0.05–0.30**: below that the anchor has
no pull, above it the model is frozen and stage 2 is pointless.

`--lambda-auto` only puts lambda in the right decade. `Q` keeps growing as training
proceeds, so the realised ratio ends up above the target — in the smoke test, asking for
0.15 gave 0.09 after one epoch and 0.34 after two.

### 3. Stage-2 runs

```bash
COMMON="--pretrained-model-path model/model_ucf.pth \
        --feature-root /path/to/UCFClipFeatures \
        --fisher-path model/fisher_ucf.pt \
        --metrics-csv model/rescale_ewc_metrics.csv"

# control: same schedule, no intervention. Never skip this one.
python ucf_train_rescale.py $COMMON --rescale-mode off --mu 1 \
  --regularizer none --lambda-reg 0 --output-model-path model/s2_ctrl.pth

# EWC only
python ucf_train_rescale.py $COMMON --rescale-mode off --mu 1 \
  --regularizer ewc --lambda-auto 0.15 --output-model-path model/s2_ewc.pth

# Eq. (1), video level
python ucf_train_rescale.py $COMMON --rescale-mode video --mu 3 \
  --regularizer ewc --lambda-auto 0.15 --output-model-path model/s2_video_mu3.pth

# Eq. (10)-(11), class level -- the variant that wins in the paper
python ucf_train_rescale.py $COMMON --rescale-mode class --mu 3 \
  --regularizer ewc --lambda-auto 0.15 --output-model-path model/s2_class_mu3.pth
```

Each run differs from `s2_ctrl` in exactly one thing. Without the control, any difference
you measure could just be "three more epochs of training".

### 3b. Adaptive weights

One pass over the train set produces the measurements; every `beta` after that is
arithmetic, so `--from-statistics` reuses them instead of re-running the model.

```bash
# measure once
python ucf_train_difficulty.py \
  --pretrained-model-path model/model_baseline_ctrl.pth \
  --feature-root /path/to/UCFClipFeatures \
  --train-list ../list/ucf_CLIP_rgb_relative.csv \
  --adaptive-beta 0.5 \
  --difficulty-output model/w_beta05.json \
  --difficulty-csv model/difficulty.csv

# the other two ablation arms, no GPU needed
python ucf_train_difficulty.py --from-statistics model/w_beta05.json \
  --adaptive-beta 0.0 --difficulty-output model/w_beta00.json
python ucf_train_difficulty.py --from-statistics model/w_beta05.json \
  --adaptive-beta 1.0 --difficulty-output model/w_beta10.json

# then train
python ucf_train_rescale.py $COMMON --rescale-mode adaptive_class \
  --class-weight-file model/w_beta05.json --output-model-path model/s2_adaptive05.pth
```

Read the Spearman line the script prints before trusting the `beta` ablation. The source
checkpoint was trained on this very train set, so a rare class got few gradient updates
and keeps a high train loss; if `|rho|` is large, "difficulty" is mostly "frequency"
restated and the two arms are not independent.

### 4. The trade-off table

```bash
python ucf_eval_perclass.py \
  --feature-root /path/to/UCFClipFeatures \
  --eval-model-paths source=model/model_ucf.pth ctrl=model/s2_ctrl.pth \
                     video_mu3=model/s2_video_mu3.pth class_mu3=model/s2_class_mu3.pth \
  --eval-output model/rescale_ewc_perclass.csv
```

Left half must not degrade, right half should improve — the layout of the paper's
Tables 3 and 4.

## Hyper-parameters

| Option | Default | Note |
|---|---|---|
| `--mu` | `1.0` | Useful range **2–10**. The paper's 100–10000 suits an unbounded CTC loss, not these O(1) losses |
| `--rescale-mode` | `class` | `off` / `video` / `class` / `adaptive_class` |
| `--class-weight-file` | — | Required by `adaptive_class`. Written by `ucf_train_difficulty.py` |
| `--adaptive-alpha` | `6.0` | Weights span `1` to `1 + alpha` |
| `--adaptive-beta` | `0.5` | `1.0` difficulty only, `0.0` frequency only. The ablation axis |
| `--adaptive-w-max` | `8.0` | Guard rail; at alpha 6 it never binds |
| `--difficulty-source` | `clasm_loss` | Which train-set measurement becomes difficulty |
| `--regularizer` | `ewc` | `none` / `l2` (Eq. 12) / `ewc` (Eq. 13) |
| `--lambda-auto` | `0` | Target `reg/task` ratio; `0.15` is a good ask |
| `--lr` | `2e-6` | 10x below stage 1, following the paper's much smaller fine-tuning lr |
| `--grad-clip` | `1.0` | The paper tightens clipping from 5 to 2 when rescaling |
| `--max-epoch` | `3` | Stage 2 is a short nudge, not a retrain |
| `--batch-size` | `64` | **Do not lower.** Paper section 5.4: a small batch can be all target samples, and the rescaled loss then explodes |
| `--target-classes` | `Explosion RoadAccidents Shooting Shoplifting` | Lowest-AUC classes with >= 20 test videos. `Abuse` scores as badly but has only 2 test videos. Under `adaptive_class` this is a **reporting** grouping only |
| `--target-oversample` | `1.0` | Counterpart of the paper's mixing ratio; off by default |

## Tests

```bash
python tests/test_losses.py             # or: python -m pytest tests/ -q
python tests/test_adaptive_weights.py
```

43 tests. The two that matter most assert that with `mu = 1` the losses here are
bit-for-bit `ucf_train.py`'s — otherwise every comparison is against a moved goalpost.
The rest cover Eq. (1) arithmetic, Eq. (10)–(11) gradient scaling (including that it
reaches non-target videos), Eq. (12)/(13), the exclusion of frozen CLIP parameters, and
that Eq. (14) squares per sample rather than per batch, and that the Fisher statistics build their quantile levels on the input's own device and dtype.

`test_adaptive_weights.py` adds the weight formula. Three of its tests are about the
experiment rather than the code: that `beta = 0` leaves the commonest class at exactly
`1.0` (the additive mix's reason to exist), that a common-but-hard class can be ranked
high by difficulty and low by frequency (the case the method exists for), and that a
per-class weight reaches the A-branch logits column by column, exactly.

## Status

Verified end to end on CPU against a synthetic UCF-shaped fixture: Fisher estimation over
12.6M trainable parameters, all four rescale modes, all three regularisers, lambda
auto-calibration, oversampling, mid-epoch and epoch-end evaluation, mAP, and the
multi-checkpoint report.

The adaptive path was checked the same way, on a 14-class synthetic set: the difficulty
pass, all four `--difficulty-source` values, the `beta` ablation via `--from-statistics`,
and every failure path (no weight file, missing file, unreadable file, a weight file that
does not cover every class). The control guarantee holds bit-for-bit — a weight file of
all `1.0` produces byte-identical trained weights to `--rescale-mode off`, while
`beta = 0.5` and `mu = 12` both diverge from it.

**No result on real UCF-Crime features yet** — that needs the GPU environment where the
features and the stage-1 checkpoint live.
