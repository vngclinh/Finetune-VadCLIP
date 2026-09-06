"""Sanity checks for UCFAugmentDataset (step 2 of the shift-consistency plan).

Run from VadCLIP/src:  python tests/test_dataset_augment.py
"""

import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import utils.tools as tools
from utils.dataset_augment import UCFAugmentDataset, shift_view, shifted_length

CLIP_DIM = 256
FEATURE_WIDTH = 512


def build_fixture(root, videos):
    """Write synthetic .npy features plus a list CSV, mirroring the real layout."""
    rows = []
    for video_id, label, snippet_count in videos:
        folder = root / label
        folder.mkdir(parents=True, exist_ok=True)
        relative = f"{label}/{video_id}__0.npy"
        # Distinct values per snippet so index alignment is verifiable.
        feature = np.tile(
            np.arange(snippet_count, dtype=np.float32).reshape(-1, 1), (1, FEATURE_WIDTH)
        )
        np.save(root / relative, feature)
        rows.append({"path": relative, "label": label, "video_id": video_id})

    list_path = root / "list.csv"
    with open(list_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "video_id"])
        writer.writeheader()
        writer.writerows(rows)
    return list_path


def test_alignment_and_shapes(root, list_path):
    label_map = {"Normal": "normal", "Abuse": "abuse"}
    offset = 26
    dataset = UCFAugmentDataset(
        CLIP_DIM, str(list_path), False, label_map, str(root),
        normal=False, shift_offset=offset, random_shift=False,
    )
    assert len(dataset) > 0, "abnormal split must not be empty"

    for index in range(len(dataset)):
        feat_full, feat_shift, label, len_full, len_shift, returned_offset = dataset[index]
        full = feat_full.numpy()
        shifted = feat_shift.numpy()

        assert full.shape == (CLIP_DIM, FEATURE_WIDTH), f"full view shape {full.shape}"
        assert shifted.shape == (CLIP_DIM, FEATURE_WIDTH), f"shifted view shape {shifted.shape}"
        assert returned_offset == offset
        assert len_shift == max(0, len_full - offset)
        assert label != "Normal", "normal=False must exclude Normal rows"

        # The core guarantee: content at j in the full view sits at j - offset after the shift.
        for j in range(offset, min(len_full, CLIP_DIM)):
            assert np.allclose(shifted[j - offset], full[j]), (
                f"misalignment at j={j} (video index {index})"
            )
        # Tail is zero padding.
        assert np.allclose(shifted[CLIP_DIM - offset:], 0.0), "tail must be zero-padded"
    print("  ok: shift alignment, shapes, lengths, Normal/Abnormal filtering")


def test_zero_offset_is_identity(root, list_path):
    label_map = {"Normal": "normal", "Abuse": "abuse"}
    dataset = UCFAugmentDataset(
        CLIP_DIM, str(list_path), False, label_map, str(root), normal=False, shift_offset=0
    )
    feat_full, feat_shift, _, len_full, len_shift, offset = dataset[0]
    assert offset == 0
    assert len_shift == len_full
    assert np.array_equal(feat_full.numpy(), feat_shift.numpy()), "offset 0 must be identity"
    print("  ok: offset 0 is the identity view")


def test_normal_split(root, list_path):
    label_map = {"Normal": "normal", "Abuse": "abuse"}
    dataset = UCFAugmentDataset(
        CLIP_DIM, str(list_path), False, label_map, str(root), normal=True, shift_offset=8
    )
    assert len(dataset) > 0
    for index in range(len(dataset)):
        assert dataset[index][2] == "Normal", "normal=True must keep only Normal rows"
    print("  ok: normal=True keeps only Normal rows")


def test_shift_view_matches_process_feat_grid():
    """Long videos get resampled first; the shift then operates on the 256 grid."""
    raw = np.tile(np.arange(1000, dtype=np.float32).reshape(-1, 1), (1, 4))
    grid, length = tools.process_feat(raw, CLIP_DIM)
    assert length == CLIP_DIM
    shifted = shift_view(grid, 26)
    assert shifted.shape == grid.shape
    assert np.allclose(shifted[:CLIP_DIM - 26], grid[26:])
    print("  ok: shift operates on the post-process_feat grid")


def test_random_offset_range(root, list_path):
    label_map = {"Normal": "normal", "Abuse": "abuse"}
    dataset = UCFAugmentDataset(
        CLIP_DIM, str(list_path), False, label_map, str(root),
        normal=False, shift_offset=26, random_shift=True,
    )
    offsets = {dataset[0][5] for _ in range(50)}
    assert offsets, "expected sampled offsets"
    assert min(offsets) >= 1 and max(offsets) <= 26, f"offsets out of range: {sorted(offsets)}"
    print(f"  ok: random offsets stay in [1, 26] (saw {len(offsets)} distinct values)")



def make_dataset(root, list_path, **kwargs):
    label_map = {"Normal": "normal", "Abuse": "abuse"}
    return UCFAugmentDataset(
        CLIP_DIM, str(list_path), False, label_map, str(root), normal=False, **kwargs
    )


def test_fixed_mode_is_unchanged(root, list_path):
    """Regression guard: the defaults must reproduce round 1 exactly.

    In particular the fixed offset deliberately ignores len_full, so the 20-snippet video
    still ends up with no overlap and drops out of the loss.
    """
    dataset = make_dataset(root, list_path, shift_offset=26)
    for index in range(len(dataset)):
        _, _, _, len_full, len_shift, offset = dataset[index]
        assert offset == 26, f"item {index}: expected offset 26, got {offset}"
        assert len_shift == max(0, len_full - 26)
    print("  ok: fixed-offset mode is byte-for-byte the original behaviour")


def test_ratio_mode_scales_with_length(root, list_path):
    """The point of ratio mode: the same dose for a 400-row video and a 20-row one."""
    dataset = make_dataset(root, list_path, shift_ratio=0.1)
    seen = {}
    for index in range(len(dataset)):
        _, _, _, len_full, _, offset = dataset[index]
        seen[len_full] = offset
        assert offset == min(round(0.1 * len_full), len_full - 1, CLIP_DIM - 1), (
            f"len {len_full}: offset {offset} is not 10% of the video"
        )
    assert len(set(seen.values())) > 1, "expected different offsets for different lengths"
    print(f"  ok: ratio mode sizes the shift per video (len -> offset: {seen})")


def test_ratio_mode_always_keeps_overlap(root, list_path):
    """The free consequence of ratio mode: no video is dropped for lack of overlap."""
    dataset = make_dataset(root, list_path, shift_ratio=0.5)
    for index in range(len(dataset)):
        _, _, _, len_full, len_shift, offset = dataset[index]
        assert offset < len_full, f"len {len_full}: offset {offset} leaves no overlap"
        assert len_shift > 0
    print("  ok: ratio mode leaves every video with an overlap, even at ratio 0.5")


def test_tail_direction_preserves_content(root, list_path):
    """A tail shift on a zero-padded video is a pure repositioning: nothing is lost."""
    dataset = make_dataset(root, list_path, shift_ratio=0.1, shift_direction="tail")
    for index in range(len(dataset)):
        feat_full, feat_shift, _, len_full, len_shift, offset = dataset[index]
        assert offset <= 0, f"tail direction produced a positive offset {offset}"
        magnitude = -offset
        if magnitude == 0:
            continue
        assert len_shift == min(CLIP_DIM, len_full + magnitude)
        # Content that still fits on the grid must survive the move untouched.
        kept = min(CLIP_DIM - magnitude, len_full)
        assert np.allclose(feat_shift[magnitude:magnitude + kept], feat_full[:kept])
        assert np.allclose(feat_shift[:magnitude], 0.0)
    print("  ok: tail shifts pad the front and carry the content across intact")


def test_both_direction_uses_both_signs(root, list_path):
    dataset = make_dataset(root, list_path, shift_ratio=0.1, shift_direction="both")
    signs = {int(np.sign(dataset[0][5])) for _ in range(60)}
    assert signs == {1, -1}, f"expected both signs across draws, saw {signs}"
    print("  ok: 'both' draws head and tail shifts per item")


def test_random_shift_in_ratio_mode(root, list_path):
    dataset = make_dataset(root, list_path, shift_ratio=0.1, random_shift=True)
    _, _, _, len_full, _, _ = dataset[0]
    ceiling = min(round(0.1 * len_full), len_full - 1, CLIP_DIM - 1)
    offsets = {dataset[0][5] for _ in range(60)}
    assert min(offsets) >= 1 and max(offsets) <= ceiling, (
        f"offsets {sorted(offsets)} outside [1, {ceiling}]"
    )
    assert len(offsets) > 1, "random_shift produced a single value"
    print(f"  ok: random magnitudes stay in [1, {ceiling}] in ratio mode")


def test_curriculum_scales_the_magnitude(root, list_path):
    dataset = make_dataset(root, list_path, shift_ratio=0.2)
    _, _, _, len_full, _, full_strength = dataset[0]
    dataset.set_shift_scale(0.5)
    _, _, _, _, _, halved = dataset[0]
    assert halved == round(0.5 * 0.2 * len_full), f"scale 0.5 gave {halved}"
    assert halved < full_strength
    dataset.set_shift_scale(1.0)
    assert dataset[0][5] == full_strength
    print(f"  ok: the curriculum scales the magnitude ({full_strength} -> {halved} -> back)")


def test_shifted_length_both_signs():
    assert shifted_length(200, 26, CLIP_DIM) == 174
    assert shifted_length(10, 26, CLIP_DIM) == 0
    assert shifted_length(200, -26, CLIP_DIM) == 226
    assert shifted_length(250, -26, CLIP_DIM) == CLIP_DIM   # clipped by the grid
    assert shifted_length(200, 0, CLIP_DIM) == 200
    print("  ok: shifted_length handles both signs and the grid ceiling")


def test_shift_view_negative_is_the_inverse():
    grid = np.tile(np.arange(CLIP_DIM, dtype=np.float32).reshape(-1, 1), (1, 4))
    back = shift_view(shift_view(grid, -13), 13)
    assert np.allclose(back[:CLIP_DIM - 13], grid[:CLIP_DIM - 13])
    print("  ok: a tail shift followed by an equal head shift restores the grid")

def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        list_path = build_fixture(root, [
            ("Abuse001_x264", "Abuse", 400),   # longer than the grid -> uniform_extract
            ("Abuse002_x264", "Abuse", 120),   # shorter than the grid -> zero padded
            ("Abuse003_x264", "Abuse", 20),    # shorter than the offset -> no overlap
            ("Normal001_x264", "Normal", 300),
        ])
        print("test_dataset_augment")
        test_alignment_and_shapes(root, list_path)
        test_zero_offset_is_identity(root, list_path)
        test_normal_split(root, list_path)
        test_shift_view_matches_process_feat_grid()
        test_random_offset_range(root, list_path)
        test_fixed_mode_is_unchanged(root, list_path)
        test_ratio_mode_scales_with_length(root, list_path)
        test_ratio_mode_always_keeps_overlap(root, list_path)
        test_tail_direction_preserves_content(root, list_path)
        test_both_direction_uses_both_signs(root, list_path)
        test_random_shift_in_ratio_mode(root, list_path)
        test_curriculum_scales_the_magnitude(root, list_path)
        test_shifted_length_both_signs()
        test_shift_view_negative_is_the_inverse()
        print("all dataset checks passed")


if __name__ == "__main__":
    main()
