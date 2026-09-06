import csv
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

import utils.tools as tools


def shift_view(feat_full, offset):
    """Translate a ``[T, D]`` view by ``offset`` grid steps, zero-padding the gap.

    The offset is **signed**, and one relation holds for both signs::

        shifted[j] = feat_full[j + offset]      whenever 0 <= j + offset < T
        shifted[j] = 0                          otherwise

    so content sitting at position ``p`` of the full view lands at ``p - offset`` of the
    returned view. Positive offsets drop the first ``offset`` rows and pad the tail
    (content moves earlier); negative offsets pad ``|offset|`` rows at the front and drop
    the same number from the tail (content moves later).

    The two directions are not equivalent in what they destroy. A positive offset always
    discards the first ``offset`` snippets of real content. A negative offset only
    discards content that is pushed past the end of the grid, so for the 72% of UCF-Crime
    videos whose valid length is below T it is a *pure* repositioning that loses nothing.
    """
    length = feat_full.shape[0]
    if offset == 0:
        return feat_full.copy()
    if abs(offset) >= length:
        raise ValueError(f"|offset| {abs(offset)} must be smaller than the grid length {length}.")
    padding = np.zeros((abs(offset), feat_full.shape[1]), dtype=feat_full.dtype)
    if offset > 0:
        return np.concatenate([feat_full[offset:], padding], axis=0)
    return np.concatenate([padding, feat_full[:offset]], axis=0)


def shifted_length(len_full, offset, clip_dim):
    """Valid length of the shifted view: how many leading grid slots the model should read.

    For a positive offset the head is cut away, so the view gets shorter. For a negative
    offset the content is pushed back behind ``|offset|`` zero rows, and those rows sit
    *inside* the prefix the model reads -- exactly what the model would see for a video
    that genuinely opens on blank snippets -- so the length grows until it hits the grid.
    """
    if offset >= 0:
        return max(0, int(len_full) - int(offset))
    return min(int(clip_dim), int(len_full) + abs(int(offset)))


class UCFAugmentDataset(data.Dataset):
    """UCF-Crime CLIP features paired with a temporally shifted second view.

    The shift is applied *after* ``tools.process_feat``, i.e. on the normalised
    ``clip_dim`` grid. Position ``j`` of the full view then corresponds exactly to
    position ``j - offset`` of the shifted view for every video, so the two views can be
    compared frame by frame without any warping. Shifting *before* ``process_feat`` would
    send the two views through ``uniform_extract`` at different compression ratios and
    break that correspondence.

    ``__getitem__`` returns ``(feat_full, feat_shift, label, len_full, len_shift, offset)``
    where ``offset`` is **signed** -- see :func:`shift_view`.

    How the offset magnitude is chosen
    ----------------------------------
    ``shift_ratio > 0`` sizes the shift as a fraction of each video's own valid length,
    ``round(shift_ratio * len_full)``; ``shift_ratio == 0`` falls back to the fixed
    ``shift_offset`` in grid steps.

    The distinction matters more than it looks. A constant offset on the 256-step grid is
    a constant fraction *of the grid*, not of the video: 72% of the training files have a
    valid length below 256 (median 138), so a fixed 26 is ~10% of the grid but ~19% of the
    median video, and for a long compressed video 26 grid steps stand for thousands of
    original feature rows. The augmentation dose then varies by orders of magnitude across
    a single batch, which is variance injected straight into the consistency term. Sizing
    by ``len_full`` makes the dose the same for every video, and as a free consequence no
    video is dropped for lack of overlap -- the fixed-offset mode loses ~3.9% of files.

    In ``test_mode`` the full view comes from ``tools.process_split`` (a ``[chunks, T, D]``
    stack) and no shifted view is produced; the offset is forced to 0 so callers that only
    need the plain test view can reuse this class.
    """

    def __init__(
        self,
        clip_dim: int,
        file_path: str,
        test_mode: bool,
        label_map: dict,
        feature_root: str,
        normal: bool = False,
        shift_offset: int = 26,
        random_shift: bool = False,
        shift_ratio: float = 0.0,
        shift_direction: str = "head",
    ):
        with open(file_path, "r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        self.feature_root = Path(feature_root)
        self.shift_offset = int(shift_offset)
        self.random_shift = bool(random_shift)
        self.shift_ratio = float(shift_ratio)
        self.shift_direction = str(shift_direction)
        # Curriculum knob: 1.0 is full strength. The trainer lowers it for the first
        # epochs when --shift-ratio-warmup is on. See set_shift_scale.
        self.shift_scale = 1.0

        if not 0 <= self.shift_offset < clip_dim:
            raise ValueError(f"shift_offset must be in [0, {clip_dim}). Got {self.shift_offset}.")
        if not 0.0 <= self.shift_ratio < 1.0:
            raise ValueError(f"shift_ratio must be in [0, 1). Got {self.shift_ratio}.")
        if self.shift_direction not in ("head", "tail", "both"):
            raise ValueError(
                f"shift_direction must be 'head', 'tail' or 'both'. Got {self.shift_direction!r}."
            )

        # Same Normal/Abnormal split as utils.dataset.UCFDataset.
        if normal and not test_mode:
            self.rows = [row for row in self.rows if row["label"] == "Normal"]
        elif not test_mode:
            self.rows = [row for row in self.rows if row["label"] != "Normal"]

        self.validate_feature_paths()

    def validate_feature_paths(self):
        missing = []
        for row in self.rows:
            feature_path = self.feature_root / row["path"]
            if not feature_path.exists():
                missing.append((row.get("video_id"), row.get("label"), row.get("path")))
        if missing:
            preview = "\n".join(f"  {video_id},{label},{path}" for video_id, label, path in missing[:50])
            extra = "" if len(missing) <= 50 else f"\n  ... and {len(missing) - 50} more"
            raise FileNotFoundError(
                "Missing feature files referenced by list.\n"
                f"feature_root: {self.feature_root}\n"
                f"missing_files: {len(missing)}\n"
                "First missing entries:\n"
                f"{preview}{extra}"
            )

    def __len__(self):
        return len(self.rows)

    def set_shift_scale(self, scale):
        """Scale every offset magnitude by ``scale`` (the ratio curriculum).

        Worker processes get a pickled copy of the dataset, so a persistent worker pool
        would never see this change. The trainer therefore turns ``persistent_workers``
        off whenever the curriculum is active.
        """
        self.shift_scale = float(scale)

    def sample_offset(self, len_full):
        """Signed offset for one item.

        ``torch.randint`` is used rather than ``random`` so DataLoader workers, which
        inherit distinct torch seeds, stay decorrelated.
        """
        if self.test_mode:
            return 0

        if self.shift_ratio > 0:
            magnitude = int(round(self.shift_ratio * self.shift_scale * int(len_full)))
            # Capping at len_full - 1 is what guarantees at least one overlapping
            # position, so no video is ever dropped from the consistency term.
            magnitude = min(magnitude, int(len_full) - 1, self.clip_dim - 1)
        else:
            # Fixed mode keeps the original semantics deliberately: the offset ignores
            # len_full, so a video shorter than it is left with no overlap and drops out
            # of the loss (3.91% of the training files at offset 26). Capping here would
            # silently change the configuration the first round was measured with.
            magnitude = min(int(round(self.shift_offset * self.shift_scale)), self.clip_dim - 1)

        if magnitude <= 0:
            return 0

        if self.random_shift:
            magnitude = int(torch.randint(1, magnitude + 1, (1,)).item())

        if self.shift_direction == "head":
            return magnitude
        if self.shift_direction == "tail":
            return -magnitude
        # 'both': draw the side per item so neither end of the video is favoured. UCF-Crime
        # events sit mostly in the middle, so a head-only shift biases which context is lost.
        return magnitude if int(torch.randint(0, 2, (1,)).item()) else -magnitude

    def __getitem__(self, index):
        row = self.rows[index]
        feature_path = self.feature_root / row["path"]
        clip_feature = np.load(feature_path)

        if not self.test_mode:
            feat_full, len_full = tools.process_feat(clip_feature, self.clip_dim)
        else:
            feat_full, len_full = tools.process_split(clip_feature, self.clip_dim)

        offset = self.sample_offset(len_full)
        if offset == 0:
            feat_shift = feat_full
            len_shift = int(len_full)
        else:
            feat_shift = shift_view(feat_full, offset)
            len_shift = shifted_length(len_full, offset, self.clip_dim)

        return (
            torch.tensor(feat_full),
            torch.tensor(feat_shift),
            row["label"],
            int(len_full),
            int(len_shift),
            int(offset),
        )
