import itertools
import logging

logger = logging.getLogger(__name__)

import os

from chainer.dataset import DatasetMixin

import imageio
import numpy as np
import tqdm

from pose.hand_dataset.dataset import HandPoseDataset as HandDataset
from pose.graphics import camera
from pose.hand_dataset.common_dataset import NUM_KEYPOINTS, STANDARD_KEYPOINT_NAMES, COLOR_MAP, EDGES, ROOT_IDX, REF_EDGE
from pose.hand_dataset.common_dataset import make_keypoint_converter, wrist2palm
from pose.hand_dataset.common_dataset import DATA_CONVENTION

SEX = ["male", "female"]
OBJ_SUFFIX = ["object", "noobject"]
SEQUENCES = ["seq01", "seq02", "seq03", "seq04", "seq05", "seq06", "seq07"]
CAMERA_DIR = ["cam01", "cam02", "cam03", "cam04", "cam05"]
PARTITIONS = ["01", "02", "03"]
MAX_FRAME_IDX = 500

DEFAULT_KEYPOINT_NAMES = [
    "W",
    "T0",
    "T1",
    "T2",
    "T3",
    "I0",
    "I1",
    "I2",
    "I3",
    "M0",
    "M1",
    "M2",
    "M3",
    "R0",
    "R1",
    "R2",
    "R3",
    "L0",
    "L1",
    "L2",
    "L3",
]

assert len(DEFAULT_KEYPOINT_NAMES) == NUM_KEYPOINTS

NAME_CONVERTER = make_keypoint_converter(
    root="W",
    fingers=["T", "I", "M", "R", "L"],
    parts=["0", "1", "2", "3"],
    sep="",
)

INDEX_CONVERTER = [DEFAULT_KEYPOINT_NAMES.index(NAME_CONVERTER[k]) for k in STANDARD_KEYPOINT_NAMES]
KEYPOINT_NAMES = STANDARD_KEYPOINT_NAMES


def create_camera_mat():
    fx = fy = 617.173
    u0 = 315.453
    v0 = 242.259
    rgb_camera_intr = camera.CameraIntr(**{"u0": u0, "v0": v0, "fx": fx, "fy": fy})
    fx = fy = 475.62
    u0 = 311.125
    v0 = 245.965
    depth_camera_intr = camera.CameraIntr(**{"u0": u0, "v0": v0, "fx": fx, "fy": fy})

    cameras = {}
    cameras["rgb_camera_intr"] = rgb_camera_intr
    cameras["depth_camera_intr"] = depth_camera_intr
    return cameras


# use color_on_depth as rgb image
USE_COLOR_ON_DEPTH = True

CAMERAS = create_camera_mat()
if USE_COLOR_ON_DEPTH:
    RGB_CAMERA_INTR = CAMERAS["depth_camera_intr"]
else:
    RGB_CAMERA_INTR = CAMERAS["rgb_camera_intr"]
DEPTH_CAMERA_INTR = CAMERAS["depth_camera_intr"]
TRANSLATION_VECTOR = np.array([24.7, -0.0471401, 3.72045])
if USE_COLOR_ON_DEPTH:
    TRANSLATION_VECTOR = np.zeros(3)


class SynthHandsBaseDataset(DatasetMixin):
    def __init__(self, dataset_dir, debug=False, mode="train"):
        self.rgb_camera = RGB_CAMERA_INTR
        self.depth_camera = DEPTH_CAMERA_INTR
        self.n_joints = NUM_KEYPOINTS
        self.root_idx = ROOT_IDX
        self.ref_edge = REF_EDGE
        self.mode = mode
        self.annotations = self.load_annotations(dataset_dir, debug=debug)

    def __len__(self):
        return len(self.annotations)

    def read_depth(self, depth_path):
        return np.expand_dims(imageio.imread(depth_path), axis=0)

    def get_example(self, i):
        return self.annotations[i]

    def load_annotations(self, dataset_dir, debug=False):
        logger.info("> load annotations mode = {}".format(self.mode))
        annotations = []
        iterable = [SEX, OBJ_SUFFIX, SEQUENCES, CAMERA_DIR, PARTITIONS]
        fmt = "{0:08d}_{1}"
        for sex, suffix, seq, cam, partition in itertools.product(*iterable):
            img_dir = os.path.join(
                dataset_dir,
                "_".join([sex, suffix]),
                seq,
                cam,
                partition,
            )

            for frame_idx in tqdm.tqdm(range(MAX_FRAME_IDX)):
                color_path = os.path.join(img_dir, fmt.format(frame_idx, "color.png"))
                color_on_depth_path = os.path.join(img_dir, fmt.format(frame_idx, "color_on_depth.png"))
                depth_path = os.path.join(img_dir, fmt.format(frame_idx, "depth.png"))
                joint_path = os.path.join(img_dir, fmt.format(frame_idx, "joint_pos.txt"))
                if not os.path.exists(color_path):
                    continue
                if not os.path.exists(depth_path):
                    continue

                world_joint = np.loadtxt(joint_path, delimiter=",").reshape(-1, 3)
                if DATA_CONVENTION == "ZYX":
                    if USE_COLOR_ON_DEPTH:
                        rgb_joint = world_joint
                    else:
                        rgb_joint = world_joint + TRANSLATION_VECTOR
                    depth_joint = world_joint
                    # flip xyz -> zyx
                    rgb_joint = rgb_joint[:, ::-1]
                    # flip xyz -> zyx
                    depth_joint = depth_joint[:, ::-1]
                else:
                    if USE_COLOR_ON_DEPTH:
                        rgb_joint = world_joint
                    else:
                        rgb_joint = world_joint + TRANSLATION_VECTOR
                    depth_joint = world_joint
                example = {}
                if USE_COLOR_ON_DEPTH:
                    example["rgb_path"] = color_on_depth_path
                else:
                    example["rgb_path"] = color_path
                rgb_joint = rgb_joint[INDEX_CONVERTER]
                rgb_joint = wrist2palm(rgb_joint)
                depth_joint = depth_joint[INDEX_CONVERTER]
                depth_joint = wrist2palm(depth_joint)
                example["hand_side"] = "left"
                example["depth_path"] = depth_path
                example["rgb_joint"] = rgb_joint
                example["depth_joint"] = depth_joint
                annotations.append(example)
                if debug:
                    break
        return annotations


def get_base_dataset(dataset_dir, **kwargs):
    base = SynthHandsBaseDataset(dataset_dir, **kwargs)
    return base


def get_synth_dataset(dataset_dir, param, **kwargs):
    base = get_base_dataset(dataset_dir, **kwargs)
    dataset = HandDataset(base, param)
    return dataset


def get_dataset(dataset_dir, param, **kwargs):
    dataset = get_synth_dataset(dataset_dir, param, **kwargs)
    return dataset


def visualize_dataset(dataset_dir):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # NOQA
    from chainercv.utils import read_image
    from pose.visualizations import vis_point, vis_edge

    dataset = SynthHandsBaseDataset(dataset_dir, debug=True, mode="train")
    example = dataset.get_example(2)
    camera_joint = example["rgb_joint"]
    depth_joint = example["depth_joint"]
    rgb_path = example["rgb_path"]
    depth_path = example["depth_path"]
    img = read_image(rgb_path)
    depth = dataset.read_depth(depth_path)
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223, projection="3d")
    ax4 = fig.add_subplot(224, projection="3d")
    rgb_vu = RGB_CAMERA_INTR.zyx2vu(camera_joint)
    color = [COLOR_MAP[k] for k in KEYPOINT_NAMES]
    edge_color = [COLOR_MAP[s, t] for s, t in EDGES]
    depth_vu = DEPTH_CAMERA_INTR.zyx2vu(depth_joint)
    vis_point(rgb_vu, img=img, color=color, ax=ax1)
    vis_edge(rgb_vu, indices=EDGES, color=edge_color, ax=ax1)

    vis_point(depth_vu, img=depth, color=color, ax=ax2)
    vis_edge(depth_vu, indices=EDGES, color=edge_color, ax=ax2)

    vis_point(camera_joint, color=color, ax=ax3)
    vis_edge(camera_joint, indices=EDGES, color=edge_color, ax=ax3)

    vis_point(depth_joint, color=color, ax=ax4)
    vis_edge(depth_joint, indices=EDGES, color=edge_color, ax=ax4)

    for ax in [ax3, ax4]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(-65, -90)

    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    dataset_dir = os.path.expanduser("~/dataset/SynthHands_Release")
    visualize_dataset(dataset_dir)
