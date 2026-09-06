"""Check that stacking the two views in the batch does not disturb the full view.

The training script feeds ``cat([full, shift])`` through a single forward pass and then
slices the full view back out. That is only valid if CLIPVAD is batch independent, which
this test verifies on the real model code. It is also the precondition for the step-5
equivalence check (``--lambda-consistency 0`` must reproduce ucf_train.py exactly).

The frozen CLIP encoder is stubbed out: encode_video never touches it, and downloading
ViT-B/16 just to test batch independence would be wasteful.

Run from VadCLIP/src:  python tests/test_two_view_batching.py
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

EMBED_DIM = 512
CONTEXT_LENGTH = 77
EOT_TOKEN = 49407
VOCAB_SIZE = 49408


class _StubCLIP(nn.Module):
    """Minimal stand-in exposing the two methods CLIPVAD calls on the frozen encoder."""

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.text_projection = nn.Parameter(torch.randn(EMBED_DIM, EMBED_DIM) * 0.01)

    def encode_token(self, token):
        return self.token_embedding(token)

    def encode_text(self, text, token):
        return text.mean(dim=1) @ self.text_projection


def install_clip_stub():
    stub = types.ModuleType("clip.clip")
    stub.load = lambda name, device: (_StubCLIP(), None)

    def tokenize(texts, truncate=False):
        tokens = torch.zeros(len(texts), CONTEXT_LENGTH, dtype=torch.long)
        tokens[:, 5] = EOT_TOKEN  # argmax picks position 5, as with a real short prompt
        return tokens

    stub.tokenize = tokenize

    package = types.ModuleType("clip")
    package.clip = stub
    sys.modules["clip"] = package
    sys.modules["clip.clip"] = stub


install_clip_stub()

from model import CLIPVAD  # noqa: E402  (import must follow the stub installation)

VISUAL_LENGTH = 256
BATCH = 3
TOLERANCE = 1e-5


def build_model():
    torch.manual_seed(234)
    model = CLIPVAD(
        num_class=14, embed_dim=EMBED_DIM, visual_length=VISUAL_LENGTH, visual_width=512,
        visual_head=1, visual_layers=2, attn_window=8,
        prompt_prefix=10, prompt_postfix=10, device="cpu",
    )
    model.eval()
    return model


def test_encode_video_is_batch_independent(model):
    full = torch.randn(BATCH, VISUAL_LENGTH, EMBED_DIM)
    shifted = torch.randn(BATCH, VISUAL_LENGTH, EMBED_DIM)
    len_full = torch.tensor([256, 200, 120])
    len_shift = torch.tensor([230, 174, 94])

    with torch.no_grad():
        alone = model.encode_video(full, None, len_full)
        together = model.encode_video(
            torch.cat([full, shifted], dim=0), None, torch.cat([len_full, len_shift], dim=0)
        )

    deviation = (alone - together[:BATCH]).abs().max().item()
    assert deviation < TOLERANCE, f"encode_video is not batch independent (max diff {deviation})"
    print(f"  ok: encode_video batch independent (max diff {deviation:.2e})")


def test_forward_is_batch_independent(model):
    prompt_text = [
        "normal", "abuse", "arrest", "arson", "assault", "burglary", "explosion",
        "fighting", "roadAccidents", "robbery", "shooting", "shoplifting", "stealing", "vandalism",
    ]
    full = torch.randn(BATCH, VISUAL_LENGTH, EMBED_DIM)
    shifted = torch.randn(BATCH, VISUAL_LENGTH, EMBED_DIM)
    len_full = torch.tensor([256, 200, 120])
    len_shift = torch.tensor([230, 174, 94])

    with torch.no_grad():
        text_alone, logits1_alone, logits2_alone = model(full, None, prompt_text, len_full)
        text_together, logits1_together, logits2_together = model(
            torch.cat([full, shifted], dim=0), None, prompt_text,
            torch.cat([len_full, len_shift], dim=0),
        )

    assert text_alone.shape == (14, EMBED_DIM), f"text features shape {text_alone.shape}"
    assert torch.equal(text_alone, text_together), "text branch must not depend on the visual batch"

    deviation1 = (logits1_alone - logits1_together[:BATCH]).abs().max().item()
    deviation2 = (logits2_alone - logits2_together[:BATCH]).abs().max().item()
    assert deviation1 < TOLERANCE, f"logits1 differs when batched (max diff {deviation1})"
    assert deviation2 < TOLERANCE, f"logits2 differs when batched (max diff {deviation2})"
    print(f"  ok: logits1 batch independent (max diff {deviation1:.2e})")
    print(f"  ok: logits2 batch independent (max diff {deviation2:.2e})")
    print("  ok: text features identical and unaffected by the extra views")


def test_shifted_input_actually_changes_scores(model):
    """Guards the premise: if the model were already shift invariant there is nothing to fix."""
    from utils.dataset_augment import shift_view

    feature = torch.randn(1, VISUAL_LENGTH, EMBED_DIM)
    shifted = torch.tensor(shift_view(feature[0].numpy(), 26)).unsqueeze(0)
    lengths = torch.tensor([256])

    with torch.no_grad():
        original = model.encode_video(feature, None, lengths)
        moved = model.encode_video(shifted, None, torch.tensor([230]))

    # Content at position j sits at j - 26 after the shift; an equivariant model would
    # produce identical features there. Random weights will not, which is the point.
    difference = (original[0, 26:] - moved[0, :VISUAL_LENGTH - 26]).abs().mean().item()
    assert difference > 0, "shifting the input should change the encoded features"
    print(f"  ok: shifting the input moves the features (mean |diff| {difference:.4f})")


def main():
    print("test_two_view_batching")
    model = build_model()
    test_encode_video_is_batch_independent(model)
    test_forward_is_batch_independent(model)
    test_shifted_input_actually_changes_scores(model)
    print("all batching checks passed")


if __name__ == "__main__":
    main()
