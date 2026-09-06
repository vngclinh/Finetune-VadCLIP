"""Sanity checks for shift_consistency_loss (step 3 of the shift-consistency plan).

The decisive test is test_true_shift_is_zero: if the index alignment were off by one, a
genuinely shifted copy would not produce a zero loss.

Run from VadCLIP/src:  python tests/test_shift_consistency_loss.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ucf_train_augment import count_without_overlap, overlap_sizes, shift_consistency_loss

BATCH = 4
LENGTH = 256
TOLERANCE = 1e-6


def shift_logits(logits, offset):
    """Same operation the dataset applies to features, but on a logit tensor.

    The offset is signed, matching utils.dataset_augment.shift_view.
    """
    if offset == 0:
        return logits.clone()
    padding = torch.zeros(logits.shape[0], abs(offset), logits.shape[2], dtype=logits.dtype)
    if offset > 0:
        return torch.cat([logits[:, offset:], padding], dim=1)
    return torch.cat([padding, logits[:, :offset]], dim=1)


def test_zero_offset_is_zero():
    for branch, channels in (("c", 1), ("a", 14)):
        logits = torch.randn(BATCH, LENGTH, channels)
        lengths = torch.full((BATCH,), LENGTH)
        loss = shift_consistency_loss(logits, logits.clone(), lengths, 0, branch=branch)
        assert loss.item() < TOLERANCE, f"branch {branch}: offset 0 gave {loss.item()}"
    print("  ok: offset 0 with identical views gives exactly 0")


def test_same_tensor_is_zero():
    for branch, channels in (("c", 1), ("a", 14)):
        logits = torch.randn(BATCH, LENGTH, channels)
        lengths = torch.full((BATCH,), LENGTH)
        loss = shift_consistency_loss(logits, logits, lengths, 0, branch=branch)
        assert loss.item() < TOLERANCE, f"branch {branch}: same tensor gave {loss.item()}"
    print("  ok: passing the same tensor twice gives 0")


def test_true_shift_is_zero():
    """A perfectly shift-equivariant model would score 0. This validates the realignment."""
    for branch, channels in (("c", 1), ("a", 14)):
        for offset in (1, 8, 26, 32):
            logits = torch.randn(BATCH, LENGTH, channels)
            shifted = shift_logits(logits, offset)
            lengths = torch.full((BATCH,), LENGTH)
            loss = shift_consistency_loss(logits, shifted, lengths, offset, branch=branch)
            assert loss.item() < TOLERANCE, (
                f"branch {branch}, offset {offset}: genuinely shifted copy gave {loss.item()}; "
                "the index alignment is wrong"
            )
    print("  ok: a genuinely shifted copy gives 0 for every offset (alignment is correct)")


def test_misalignment_is_detected():
    """Off-by-one in the alignment must not pass unnoticed."""
    logits = torch.randn(BATCH, LENGTH, 1)
    shifted = shift_logits(logits, 26)
    lengths = torch.full((BATCH,), LENGTH)
    wrong = shift_consistency_loss(logits, shifted, lengths, 25, branch="c")
    assert wrong.item() > TOLERANCE, "using the wrong offset should produce a non-zero loss"
    print(f"  ok: wrong offset is detected (loss={wrong.item():.6f})")


def test_per_sample_offsets():
    channels = 1
    offsets = torch.tensor([4, 12, 26, 7])
    logits = torch.randn(BATCH, LENGTH, channels)
    shifted = torch.stack(
        [shift_logits(logits[i: i + 1], int(offsets[i]))[0] for i in range(BATCH)], dim=0
    )
    lengths = torch.full((BATCH,), LENGTH)
    loss = shift_consistency_loss(logits, shifted, lengths, offsets, branch="c")
    assert loss.item() < TOLERANCE, f"per-sample offsets gave {loss.item()}"
    print("  ok: per-sample offsets (random-shift mode) align correctly")


def test_videos_without_overlap_are_dropped():
    """A video shorter than the offset must not contribute to the average."""
    offset = 26
    lengths = torch.tensor([LENGTH, 10, LENGTH, 5])  # entries 1 and 3 have no overlap
    logits = torch.randn(BATCH, LENGTH, 1)
    noise = torch.randn(BATCH, LENGTH, 1)

    loss_all = shift_consistency_loss(logits, noise, lengths, offset, branch="c")
    # Same computation keeping only the two videos that do have overlap.
    keep = torch.tensor([0, 2])
    loss_kept = shift_consistency_loss(
        logits[keep], noise[keep], lengths[keep], offset, branch="c"
    )
    assert torch.allclose(loss_all, loss_kept, atol=1e-6), (
        f"short videos leaked into the average: {loss_all.item()} vs {loss_kept.item()}"
    )
    assert count_without_overlap(lengths, offset, BATCH) == 2
    print("  ok: videos with len_full <= offset are dropped and counted")


def test_detach_anchor_blocks_gradient():
    logits_full = torch.randn(BATCH, LENGTH, 1, requires_grad=True)
    logits_shift = torch.randn(BATCH, LENGTH, 1, requires_grad=True)
    lengths = torch.full((BATCH,), LENGTH)

    loss = shift_consistency_loss(logits_full, logits_shift, lengths, 26, branch="c", detach_anchor=True)
    loss.backward()
    assert logits_full.grad is None or torch.all(logits_full.grad == 0), (
        "detach_anchor=True must not send gradient into the full view"
    )
    assert logits_shift.grad is not None and torch.any(logits_shift.grad != 0)
    print("  ok: detach_anchor freezes the full view as a target")


def test_symmetric_kl_is_symmetric():
    logits_a = torch.randn(BATCH, LENGTH, 14)
    logits_b = torch.randn(BATCH, LENGTH, 14)
    lengths = torch.full((BATCH,), LENGTH)
    forward = shift_consistency_loss(logits_a, logits_b, lengths, 0, branch="a")
    backward = shift_consistency_loss(logits_b, logits_a, lengths, 0, branch="a")
    assert torch.allclose(forward, backward, atol=1e-6), "branch 'a' must be symmetric"
    assert forward.item() > 0, "different distributions must produce a positive KL"
    print(f"  ok: branch 'a' KL is symmetric (loss={forward.item():.6f})")


def test_gradient_flows():
    logits_full = torch.randn(BATCH, LENGTH, 1, requires_grad=True)
    logits_shift = torch.randn(BATCH, LENGTH, 1, requires_grad=True)
    lengths = torch.full((BATCH,), LENGTH)
    loss = shift_consistency_loss(logits_full, logits_shift, lengths, 26, branch="c")
    loss.backward()
    assert torch.any(logits_full.grad != 0) and torch.any(logits_shift.grad != 0)
    # Padding positions outside the overlap must receive no gradient.
    assert torch.all(logits_full.grad[:, :26] == 0), "positions before the offset must be masked"
    print("  ok: gradient flows to both views and only inside the overlap")



def test_negative_offset_alignment_is_zero():
    """Tail shifts must realign as exactly as head shifts, or the sign handling is wrong."""
    for branch, channels in (("c", 1), ("a", 14)):
        for offset in (-1, -8, -26, -32):
            logits = torch.randn(BATCH, LENGTH, channels)
            shifted = shift_logits(logits, offset)
            lengths = torch.full((BATCH,), LENGTH)
            loss = shift_consistency_loss(logits, shifted, lengths, offset, branch=branch)
            assert loss.item() < TOLERANCE, (
                f"branch {branch}, offset {offset}: genuinely shifted copy gave {loss.item()}"
            )
    print("  ok: negative (tail) offsets realign correctly too")


def test_mixed_sign_offsets_align():
    """--shift-direction both puts the two signs in one batch."""
    offsets = torch.tensor([12, -12, 26, -5])
    logits = torch.randn(BATCH, LENGTH, 1)
    shifted = torch.stack(
        [shift_logits(logits[i: i + 1], int(offsets[i]))[0] for i in range(BATCH)], dim=0
    )
    lengths = torch.full((BATCH,), LENGTH)
    loss = shift_consistency_loss(logits, shifted, lengths, offsets, branch="c")
    assert loss.item() < TOLERANCE, f"mixed-sign offsets gave {loss.item()}"
    print("  ok: a batch mixing head and tail shifts aligns correctly")


def test_negative_offset_never_drops_a_video():
    """The reason to shift toward the tail: short videos keep an overlap."""
    lengths = torch.tensor([LENGTH, 10, LENGTH, 5])
    assert count_without_overlap(lengths, 26, BATCH, grid_length=LENGTH) == 2
    assert count_without_overlap(lengths, -26, BATCH, grid_length=LENGTH) == 0
    print("  ok: head shift drops the 2 short videos, tail shift drops none")


def test_overlap_sizes_match_the_definition():
    lengths = torch.tensor([LENGTH, 100, 30, 200])
    for offset in (0, 5, 26, 120, -5, -26, -120):
        sizes = overlap_sizes(lengths, offset, LENGTH, BATCH)
        for i, length in enumerate(lengths.tolist()):
            expected = sum(
                1 for p in range(length) if 0 <= p - offset < LENGTH
            )
            assert int(sizes[i]) == expected, (
                f"offset {offset}, len {length}: got {int(sizes[i])}, expected {expected}"
            )
    print("  ok: overlap sizes match a brute-force count for both signs")

def main():
    torch.manual_seed(0)
    print("test_shift_consistency_loss")
    test_zero_offset_is_zero()
    test_same_tensor_is_zero()
    test_true_shift_is_zero()
    test_misalignment_is_detected()
    test_per_sample_offsets()
    test_videos_without_overlap_are_dropped()
    test_negative_offset_alignment_is_zero()
    test_mixed_sign_offsets_align()
    test_negative_offset_never_drops_a_video()
    test_overlap_sizes_match_the_definition()
    test_detach_anchor_blocks_gradient()
    test_symmetric_kl_is_symmetric()
    test_gradient_flows()
    print("all loss checks passed")


if __name__ == "__main__":
    main()
