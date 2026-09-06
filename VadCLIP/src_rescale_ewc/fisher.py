"""Diagonal Fisher information for EWC, after Eq. (14) of arXiv:2302.09723.

                1              ( d L(d, theta') )^2
    F_i  =  ------- *  sum     ( -------------- )
              |D|     d in D   (    d theta'_i  )

Read that literally: **square the per-sample gradient, then average**. Averaging the
gradients first and squaring afterwards is a different quantity -- opposite-signed
gradients cancel and F comes out systematically too small -- so the estimator below runs
one sample at a time. The paper evaluates it on the source dataset at the converged
source weights, which is what makes the Hessian-to-squared-gradient approximation valid
in the first place.

No CLIP import here, so this file is unit-testable on its own.
"""

import torch


def trainable_named_parameters(model):
    """VadCLIP freezes the whole CLIP backbone, so this is far smaller than the model."""
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


def clone_anchor(model, device=None):
    """theta' -- a detached copy of the converged source weights."""
    anchor = {}
    for name, parameter in trainable_named_parameters(model):
        value = parameter.detach().clone()
        anchor[name] = value.to(device) if device is not None else value
    return anchor


def zero_fisher(model):
    return {name: torch.zeros_like(p, dtype=torch.float32)
            for name, p in trainable_named_parameters(model)}


def accumulate_squared_gradients(model, loss, fisher):
    """One term of the sum in Eq. (14): backward once, add the squared gradient.

    ``allow_unused`` is on because L3 only touches the text prompt embeddings, so on a
    given sample some parameters legitimately receive no gradient.
    """
    names = [name for name, _ in trainable_named_parameters(model)]
    parameters = [p for _, p in trainable_named_parameters(model)]
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    for name, gradient in zip(names, gradients):
        if gradient is not None:
            fisher[name] += gradient.detach().float().pow(2)
    return fisher


def finalize_fisher(fisher, count):
    return {name: value / max(1, count) for name, value in fisher.items()}


def normalize_fisher(fisher, mode="mean"):
    """Rescale F so that lambda becomes a portable, O(1)-tuneable number.

    The paper's lambda = 1e7 / 5e7 only means anything at the gradient scale of CTC on
    LibriSpeech; copying it to another model is meaningless. Dividing F by its own mean
    puts the EWC penalty on the same scale as the plain L2 penalty of Eq. (12), so the
    same lambda can be compared across the two regularisers -- which is exactly the
    comparison the paper's Tables 3 and 4 make.
    """
    if mode == "none":
        return fisher
    flat = torch.cat([value.reshape(-1) for value in fisher.values()])
    if mode == "mean":
        denominator = flat.mean()
    elif mode == "max":
        denominator = flat.max()
    else:
        raise ValueError(f"mode must be 'none', 'mean' or 'max'. Got {mode!r}.")
    denominator = denominator.clamp(min=1e-20)
    return {name: value / denominator for name, value in fisher.items()}


QUANTILE_LEVELS = (0.5, 0.9, 0.99, 0.999)

# torch.quantile refuses inputs beyond 2**24 elements, so anything larger is sampled down
# to that. VadCLIP has 12.6M trainable parameters and stays under it, but the Fisher code
# should not depend on that staying true.
_QUANTILE_INPUT_LIMIT = 2 ** 24


def _quantiles(flat):
    """The quantile levels of ``flat``, computed on whatever device it already lives on."""
    values = flat
    if values.numel() > _QUANTILE_INPUT_LIMIT:
        index = torch.randint(0, values.numel(), (_QUANTILE_INPUT_LIMIT,), device=values.device)
        values = values[index]
    levels = torch.tensor(QUANTILE_LEVELS, device=values.device, dtype=values.dtype)
    return [float(value) for value in torch.quantile(values, levels)]


def fisher_statistics(fisher):
    """Numbers to sanity-check F before trusting it, and to pick lambda from.

    Expect a very heavy tail. If the median and the maximum are close, F is nearly flat and
    Eq. (13) has degenerated into Eq. (12) -- EWC would then buy nothing over plain L2.
    """
    flat = torch.cat([value.reshape(-1) for value in fisher.values()]).float()
    statistics = {
        "num_parameters": int(flat.numel()),
        "mean": float(flat.mean()),
        "max": float(flat.max()),
        "min": float(flat.min()),
        "zero_fraction": float((flat == 0).float().mean()),
    }
    # Never let a reporting failure destroy an expensive estimation run.
    try:
        median, p90, p99, p999 = _quantiles(flat)
        statistics.update({"median": median, "p90": p90, "p99": p99, "p999": p999})
    except RuntimeError as error:
        statistics["quantiles_unavailable"] = str(error)
    return statistics


def module_importance(fisher, top_k=15):
    """Mean F per top-level module, so you can see which blocks the source task relies on."""
    totals = {}
    counts = {}
    for name, value in fisher.items():
        module = name.split(".")[0]
        totals[module] = totals.get(module, 0.0) + float(value.sum())
        counts[module] = counts.get(module, 0) + value.numel()
    rows = [(module, totals[module] / max(1, counts[module]), counts[module]) for module in totals]
    rows.sort(key=lambda row: row[1], reverse=True)
    return rows[:top_k]


def save_fisher(path, fisher, anchor, meta):
    torch.save({"fisher": fisher, "anchor": anchor, "meta": meta}, path)


def load_fisher(path, device, normalize="mean"):
    """Returns ``(fisher, anchor, meta)``; ``fisher`` is None when the file holds no F.

    A file written with ``--regularizer l2`` carries only the anchor, which is all
    Eq. (12) needs.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    anchor = {name: value.to(device) for name, value in payload["anchor"].items()}
    fisher = payload.get("fisher")
    if fisher is not None:
        fisher = {name: value.to(device) for name, value in fisher.items()}
        fisher = normalize_fisher(fisher, normalize)
    return fisher, anchor, payload.get("meta", {})
