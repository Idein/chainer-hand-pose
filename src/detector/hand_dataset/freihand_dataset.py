import logging

logger = logging.getLogger(__name__)

import json
import os

from chainer.dataset import DatasetMixin
import numpy as np
import tqdm

from detector.graphics import camera
from detector.hand_dataset.common_dataset import NUM_KEYPOINTS, STANDARD_KEYPOINT_NAMES, COLOR_MAP, EDGES, ROOT_IDX, REF_EDGE
from detector.hand_dataset.common_dataset import make_keypoint_converter, wrist2palm
from detector.hand_dataset.bbox_dataset import HandBBoxDataset as HandDataset
from detector.hand_dataset.geometry_utils import DATA_CONVENTION

"""
Keypoints available:
0: left wrist, 1-4: left thumb [tip to palm], 5-8: left index, ..., 17-20: left pinky,
21: right wrist, 22-25: right thumb, ..., 38-41: right pinky
"""

DEFAULT_KEYPOINT_NAMES = [
    "wrist",
    "thumb_tip",
    "thumb_dip",
    "thumb_pip",
    "thumb_mcp",
    "index_tip",
    "index_dip",
    "index_pip",
    "index_mcp",
    "middle_tip",
    "middle_dip",
    "middle_pip",
    "middle_mcp",
    "ring_tip",
    "ring_dip",
    "ring_pip",
    "ring_mcp",
    "pinky_tip",
    "pinky_dip",
    "pinky_pip",
    "pinky_mcp",
]

NAME_CONVERTER = make_keypoint_converter(
    root="wrist",
    fingers=["thumb", "index", "middle", "ring", "pinky"],
    parts=["mcp", "pip", "dip", "tip"],
)

INDEX_CONVERTER = [DEFAULT_KEYPOINT_NAMES.index(NAME_CONVERTER[k]) for k in STANDARD_KEYPOINT_NAMES]
KEYPOINT_NAMES = STANDARD_KEYPOINT_NAMES

VAL_LIST = []


class FHBaseDataset(DatasetMixin):
    def __init__(self, dataset_dir, debug=False, mode="train"):
        self.mode = mode
        self.rgb_camera = None
        self.depth_camera = None
        self.n_joints = NUM_KEYPOINTS
        self.annotations = self.load_annotations(dataset_dir, debug=debug)
        self.root_idx = ROOT_IDX  # will be deprecated in the future
        self.ref_edge = REF_EDGE  # will be deprecated in the future

    def __len__(self):
        return len(self.annotations)

    def get_example(self, i):
        return self.annotations[i]

    def load_annotations(self, dataset_dir, debug=False):
        logger.info("load annotations mode = {}".format(self.mode))
        logger.info("dataset_dir = {}".format(dataset_dir))
        annotations = []
        with open(os.path.join(dataset_dir, "training_xyz.json"), 'r') as f:
            joint_annotations = json.load(f)
        with open(os.path.join(dataset_dir, "training_K.json"), 'r') as f:
            camera_annotations = json.load(f)
        global VAL_LIST
        VAL_LIST = np.random.choice(len(joint_annotations), int(0.1 * len(joint_annotations)))
        for idx in range(len(joint_annotations)):
            if self.mode == "train" and idx in VAL_LIST:
                continue
            if self.mode != "train" and idx not in VAL_LIST:
                continue
            rgb_joint = np.array(joint_annotations[idx])
            [
                [fx, _, u0],
                [_, fy, v0],
                [_, _, _]
            ] = camera_annotations[idx]
            rgb_camera = camera.CameraIntr(u0=u0, v0=v0, fx=fx, fy=fy)
            if DATA_CONVENTION == "ZYX":
                rgb_joint = rgb_joint[:, ::-1]
                rgb_joint = rgb_joint[INDEX_CONVERTER]
            rgb_path = os.path.join(dataset_dir, "training", "rgb", "{:08d}.jpg".format(idx))
            rgb_joint = rgb_joint[INDEX_CONVERTER]
            rgb_joint = wrist2palm(rgb_joint)
            example = {}
            example["hand_side"] = "right"
            example["rgb_path"] = rgb_path
            example["rgb_joint"] = 1000 * rgb_joint
            example["rgb_camera"] = rgb_camera
            annotations.append(example)
            if debug and len(annotations) > 10:
                break
        return annotations


def get_base_dataset(dataset_dir, **kwargs):
    base = FHBaseDataset(dataset_dir, **kwargs)
    return base


def get_fh_dataset(dataset_dir, param, **kwargs):
    base = get_base_dataset(dataset_dir, **kwargs)
    dataset = HandDataset(base, param)
    return dataset


def get_dataset(dataset_dir, param, **kwargs):
    dataset = get_fh_dataset(dataset_dir, param, **kwargs)
    return dataset


def visualize_dataset(dataset_dir):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # NOQA
    from chainercv.utils import read_image
    from .visualizations import vis_point, vis_edge

    dataset = FHBaseDataset(dataset_dir, debug=True, mode="train")
    idx = np.random.choice(len(dataset))
    example = dataset.get_example(idx)
    camera_joint = example["rgb_joint"]
    rgb_path = example["rgb_path"]
    img = read_image(rgb_path)
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(221)
    ax3 = fig.add_subplot(223, projection="3d")

    rgb_vu = example["rgb_camera"].zyx2vu(camera_joint)
    color = [COLOR_MAP[k] for k in STANDARD_KEYPOINT_NAMES]
    edge_color = [COLOR_MAP[s, t] for s, t in EDGES]
    vis_point(rgb_vu, img=img, color=color, ax=ax1)
    vis_edge(rgb_vu, indices=EDGES, color=edge_color, ax=ax1)

    vis_point(camera_joint, color=color, ax=ax3)
    vis_edge(camera_joint, indices=EDGES, color=edge_color, ax=ax3)

    for ax in [ax3]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(-65, -90)

    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    dataset_dir = os.path.expanduser("~/dataset/FreiHAND_pub_v1")
    visualize_dataset(dataset_dir)
