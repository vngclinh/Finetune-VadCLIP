import json
import csv
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

import utils.tools as tools


def load_description_map(description_json):
    with open(description_json, "r", encoding="utf-8") as f:
        records = json.load(f)

    description_map = {}
    for record in records:
        video_id = record.get("video_id")
        description = record.get("gpt_description") or record.get("openai_description") or ""
        if video_id and description.strip():
            description_map[video_id] = description.strip()
    return description_map


class UCFDescriptionDataset(data.Dataset):
    def __init__(
        self,
        clip_dim: int,
        file_path: str,
        test_mode: bool,
        label_map: dict,
        feature_root: str,
        description_json: str,
        normal: bool = False,
        require_description: bool = True,
    ):
        with open(file_path, "r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        self.feature_root = Path(feature_root)
        self.description_map = load_description_map(description_json)
        self.require_description = require_description

        if normal and not test_mode:
            self.rows = [row for row in self.rows if row["label"] == "Normal"]
        elif not test_mode:
            self.rows = [row for row in self.rows if row["label"] != "Normal"]

        if require_description:
            missing = sorted({row["video_id"] for row in self.rows} - set(self.description_map))
            if missing:
                preview = ", ".join(missing[:10])
                raise ValueError(f"Missing descriptions for {len(missing)} videos. First missing: {preview}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        feature_path = self.feature_root / row["path"]
        clip_feature = np.load(feature_path)

        if not self.test_mode:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        video_id = row["video_id"]
        description = self.description_map.get(video_id, "")
        if self.require_description and not description:
            raise KeyError(f"Missing description for video_id={video_id}")

        clip_feature = torch.tensor(clip_feature)
        clip_label = row["label"]
        return clip_feature, clip_label, clip_length, video_id, description
