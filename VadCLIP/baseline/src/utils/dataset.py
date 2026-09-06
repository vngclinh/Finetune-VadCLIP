import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import os
import utils.tools as tools

def resolve_feature_path(path, feature_root):
    """ADDED: join a list entry against --feature-root.

    Upstream ships CSVs holding the absolute paths of the author's machine
    (/home/xbgydx/Desktop/UCFClipFeatures/...). Passing feature_root='' keeps that
    behaviour byte for byte; passing a root makes the relative CSVs usable instead.
    Absolute entries always win, so a mixed list still works.
    """
    if not feature_root or os.path.isabs(path):
        return path
    return os.path.join(feature_root, path)

class UCFDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict, normal: bool = False, feature_root: str = ''):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        self.feature_root = feature_root
        if normal == True and test_mode == False:
            self.df = self.df.loc[self.df['label'] == 'Normal']
            self.df = self.df.reset_index()
        elif test_mode == False:
            self.df = self.df.loc[self.df['label'] != 'Normal']
            self.df = self.df.reset_index()
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(resolve_feature_path(self.df.loc[index]['path'], self.feature_root))
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length

class XDDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict, feature_root: str = ''):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.feature_root = feature_root
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(resolve_feature_path(self.df.loc[index]['path'], self.feature_root))
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length
