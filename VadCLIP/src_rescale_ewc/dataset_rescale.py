"""UCF-Crime feature dataset for stage-2 runs.

``VadCLIP/src/utils/dataset.py`` is the pristine upstream copy and reads the absolute
paths baked into the author's CSVs, so it cannot be pointed at a feature directory. This
class is the same dataset with a ``feature_root`` prefix and an up-front existence check,
following ``utils/dataset_augment.py``'s conventions. The feature processing itself is
still ``utils.tools.process_feat`` / ``process_split``, untouched.

It also exposes ``labels``, which the target-class oversampler needs.
"""

import csv
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

import utils.tools as tools


class UCFRescaleDataset(data.Dataset):
    def __init__(self, clip_dim, file_path, test_mode, feature_root, normal=False):
        with open(file_path, "r", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))

        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.feature_root = Path(feature_root)

        # The same Normal/Abnormal split as utils.dataset.UCFDataset.
        if normal and not test_mode:
            self.rows = [row for row in self.rows if row["label"] == "Normal"]
        elif not test_mode:
            self.rows = [row for row in self.rows if row["label"] != "Normal"]

        self.validate_feature_paths()

    @property
    def labels(self):
        return [row["label"] for row in self.rows]

    def validate_feature_paths(self):
        missing = [(row.get("video_id"), row.get("label"), row.get("path"))
                   for row in self.rows if not (self.feature_root / row["path"]).exists()]
        if missing:
            preview = "\n".join(f"  {video_id},{label},{path}" for video_id, label, path in missing[:50])
            extra = "" if len(missing) <= 50 else f"\n  ... and {len(missing) - 50} more"
            raise FileNotFoundError(
                "Missing feature files referenced by list.\n"
                f"feature_root: {self.feature_root}\n"
                f"missing_files: {len(missing)}\n"
                f"First missing entries:\n{preview}{extra}"
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        clip_feature = np.load(self.feature_root / row["path"])
        if self.test_mode:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        return torch.tensor(clip_feature), row["label"], clip_length
