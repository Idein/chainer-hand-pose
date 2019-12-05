"""
ICVL dataset this module is not used for our purpose.
"""

import logging

logger = logging.getLogger(__name__)

import os

from chainer.dataset import DatasetMixin

import imageio
import numpy as  np
import pandas as pd

from pose.hand_dataset.common_dataset import DATA_CONVENTION

from pose.utils import pairwise

FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
NUM_KEYPOINTS = 16

KEYPOINT_NAMES = ["Palm"]
for f in FINGERS:
    for p in ["root", "mid", "tip"]:
        name = "_".join([f, p])
        KEYPOINT_NAMES.append(name)
assert len(KEYPOINT_NAMES) == NUM_KEYPOINTS

EDGE_NAMES = []
for f in FINGERS:
    for s, t in pairwise(["Palm", "root", "mid", "tip"]):
        s_name = "_".join([f, s]) if s != "Palm" else s
        t_name = "_".join([f, t])
        EDGE_NAMES.append([s_name, t_name])

EDGES = [(KEYPOINT_NAMES.index(s), KEYPOINT_NAMES.index(t))
         for s, t in EDGE_NAMES]

DF_COLUMNS = ["file_path"]
for k in KEYPOINT_NAMES:
    for c in ["x", "y", "z"]:
        c_k = "_".join([c, k])
        DF_COLUMNS.append(c_k)


class ICVLBaseDataset(DatasetMixin):
    def __init__(self, dataset_dir, debug=False, mode="train"):
        self.mode = mode
        self.annotations = self.load_annotations(dataset_dir, debug=debug)

    def __len__(self):
        return len(self.annotations)

    def read_depth(self, depth_path):
        return np.expand_dims(imageio.imread(depth_path), axis=0)

    def get_example(self, i):
        return self.annotations[i]

    def load_annotations(self, dataset_dir, debug):
        annotations = []
        df = pd.read_csv(os.path.join(dataset_dir, "labels.txt"), header=None, sep=' ')
        df.columns = DF_COLUMNS
        num_data = df.shape[0]
        for idx in range(num_data):
            s = df.iloc[idx]
            depth_path = os.path.join(dataset_dir, "Depth", s["file_path"])
            depth_joint = s[1:].values.reshape(-1, 3).astype(float)
            if DATA_CONVENTION == "ZYX":
                depth_joint = depth_joint[:, ::-1]
            example = {}
            example["depth_path"] = depth_path
            example["depth_joint"] = depth_joint
            annotations.append(example)
            if debug:
                break
