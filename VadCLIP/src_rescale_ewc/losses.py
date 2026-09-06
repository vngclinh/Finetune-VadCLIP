"""Loss rescaling + weight consolidation for VadCLIP, after arXiv:2302.09723.

The paper ("Emphasizing Unseen Words", Qu et al. 2023) fine-tunes an ASR model on rare
words with two complementary interventions, both of which are reproduced here for the
UCF-Crime anomaly classes:

* **Loss rescaling** so the model stops treating every training unit equally.
  Eq. (1) multiplies the loss of a whole *utterance* containing a target word by ``mu``;
  Eq. (10)-(11) instead multiply only the *gradient* reaching the target word's nodes,
  leaving the loss value untouched. The second form wins on both axes in the paper's
  Table 4, because scaling a whole sample also amplifies whatever noise it contains.
* **Weight consolidation** so the fine-tuned model does not forget the source task.
  Eq. (12) is plain L2 towards the source weights; Eq. (13) weighs each parameter by the
  diagonal Fisher information ``F_i`` estimated at the converged source model.

Nothing here imports CLIP or ``model.py``, so the whole file is unit-testable on CPU.
The unweighted paths are numerically identical to ``ucf_train.py``'s ``CLAS2``/``CLASM``;
``tests/test_losses.py`` asserts that.
"""

import torch
import torch.nn.functional as F


# --- Target-set bookkeeping ----------------------------------------------------------

def build_video_weights(raw_labels, target_classes, mu, device):
    """Eq. (1): ``w_i = mu`` when video ``i`` belongs to a target class, else ``1``.

    ``raw_labels`` are the CSV label strings ('Explosion', 'Normal', ...) in batch order.
    """
    weights = [mu if label in target_classes else 1.0 for label in raw_labels]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_class_scale(prompt_text, label_map, target_classes, mu, device):
    """Eq. (10)-(11): a ``[num_classes]`` gradient multiplier, ``mu`` on target columns.

    ``prompt_text`` is the ordered list of prompt strings that indexes the A-branch
    logits; ``label_map`` maps a raw CSV label onto its prompt string.
    """
    scale = torch.ones(len(prompt_text), dtype=torch.float32, device=device)
    for raw_label in target_classes:
        if raw_label not in label_map:
            raise KeyError(f"Target class {raw_label!r} is not in the label map.")
        prompt = label_map[raw_label]
        if prompt not in prompt_text:
            raise KeyError(f"Prompt {prompt!r} for target class {raw_label!r} is not in prompt_text.")
        scale[prompt_text.index(prompt)] = mu
    return scale


class ScaleClassGrad(torch.autograd.Function):
    """Identity forward, per-column gradient rescaling backward.

    This is the VadCLIP counterpart of Eq. (10)-(11). The paper does not recompute the
    CTC loss after rescaling; it multiplies the gradient arriving at the OOV nodes by
    ``mu``, so the reported loss is unchanged and only the update direction shifts. Same
    here: the value of ``CLASM`` is untouched, but the gradient reaching the target
    classes' logits is multiplied by ``mu``.
    """

    @staticmethod
    def forward(ctx, logits, scale):
        ctx.save_for_backward(scale)
        return logits.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (scale,) = ctx.saved_tensors
        return grad_output * scale, None


def _reduce(per_video, weights, normalize):
    """Turn per-video losses into a scalar.

    ``normalize='none'`` divides by the batch size, which is what Eq. (1) does: the
    rescaled losses are averaged over the mini-batch exactly as before, so the total
    gradient magnitude grows with ``mu``. That is deliberate in the paper and is why it
    pairs rescaling with a much smaller learning rate and tighter gradient clipping.
    ``normalize='mean'`` divides by the sum of weights instead, holding the loss scale
    fixed; it is offered as a safer variant, not as the faithful one.
    """
    if weights is None:
        return per_video.mean()
    weighted = per_video * weights
    if normalize == "mean":
        return weighted.sum() / weights.sum().clamp(min=1e-8)
    if normalize == "none":
        return weighted.mean()
    raise ValueError(f"normalize must be 'none' or 'mean'. Got {normalize!r}.")


# --- The two VadCLIP losses, with optional rescaling ---------------------------------
#
# The top-k selection, the k = len/16 + 1 rule and the label handling are copied verbatim
# from ucf_train.py. The only change is that the final reduction is per-video first, so a
# weight can be applied before averaging.

def CLAS2(logits, labels, lengths, device, weights=None, normalize="none"):
    """C-branch binary MIL loss. ``weights=None`` reproduces ``ucf_train.CLAS2`` exactly."""
    instance_logits = torch.zeros(0).to(device)
    targets = 1 - labels[:, 0].reshape(labels.shape[0])
    targets = targets.to(device)
    probabilities = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(probabilities.shape[0]):
        top, _ = torch.topk(probabilities[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        instance_logits = torch.cat([instance_logits, torch.mean(top).view(1)], dim=0)

    per_video = F.binary_cross_entropy(instance_logits, targets, reduction="none")
    return _reduce(per_video, weights, normalize)


def CLASM(logits, labels, lengths, device, weights=None, normalize="none", class_scale=None):
    """A-branch MIL cross-entropy over the 14 classes.

    ``weights`` applies the Eq. (1) video-level rescaling; ``class_scale`` applies the
    Eq. (10)-(11) gradient-level rescaling. They are independent and can be combined,
    though the experiment grid uses one at a time.

    With both left at ``None`` this reproduces ``ucf_train.CLASM`` exactly.
    """
    instance_logits = torch.zeros(0).to(device)
    normalized_labels = labels / torch.sum(labels, dim=1, keepdim=True)
    normalized_labels = normalized_labels.to(device)

    for i in range(logits.shape[0]):
        top, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(top, 0, keepdim=True)], dim=0)

    if class_scale is not None:
        instance_logits = ScaleClassGrad.apply(instance_logits, class_scale)

    per_video = -torch.sum(normalized_labels * F.log_softmax(instance_logits, dim=1), dim=1)
    return _reduce(per_video, weights, normalize)


def text_separation_loss(text_features, device):
    """L3, copied verbatim from ucf_train.py. Never rescaled: it has no video dimension."""
    loss = torch.zeros(1).to(device)
    text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
    for j in range(1, text_features.shape[0]):
        text_feature_abnormal = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
        loss += torch.abs(text_feature_normal @ text_feature_abnormal)
    return loss / 13 * 1e-1


# --- Weight consolidation ------------------------------------------------------------

def consolidation_penalty(model, anchor, fisher, lam):
    """Eq. (12) / Eq. (13), depending on whether ``fisher`` is given.

        fisher is None : (lam/2) * sum_i (theta_i - theta'_i)^2            -> L2, Eq. (12)
        fisher given   : (lam/2) * sum_i F_i * (theta_i - theta'_i)^2      -> EWC, Eq. (13)

    ``anchor`` holds the converged source weights ``theta'``; both dicts are keyed by
    ``model.named_parameters()`` names and only cover trainable parameters. ``fisher`` is
    expected to be already normalised (see ``fisher.normalize_fisher``), because the
    paper's lambda values only make sense at the gradient scale they were tuned on.
    """
    device = next(model.parameters()).device
    if lam == 0 or not anchor:
        return torch.zeros((), device=device)

    total = torch.zeros((), device=device)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or name not in anchor:
            continue
        squared_drift = (parameter - anchor[name]).pow(2)
        if fisher is None:
            total = total + squared_drift.sum()
        else:
            importance = fisher.get(name)
            if importance is None:
                continue
            total = total + (importance * squared_drift).sum()
    return 0.5 * lam * total


@torch.no_grad()
def consolidation_scale(model, anchor, fisher):
    """The bare quadratic form ``sum_i F_i (theta_i - theta'_i)^2``, without lambda.

    Used to solve for lambda instead of hand-searching it across five orders of magnitude
    the way the paper does. Given a target ratio r and the current task loss L,
    ``lambda = 2 * r * L / this``, since the penalty is ``(lambda/2) * this``.
    """
    return float(2.0 * consolidation_penalty(model, anchor, fisher, lam=1.0).detach())


@torch.no_grad()
def mean_absolute_drift(model, anchor):
    """``mean |theta - theta'|`` over trainable parameters. Diagnostic only."""
    if not anchor:
        return 0.0
    total = 0.0
    count = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or name not in anchor:
            continue
        total += (parameter - anchor[name]).abs().sum().item()
        count += parameter.numel()
    return total / max(1, count)
