import logging

logger = logging.getLogger(__name__)

import os
import pickle

from chainer.dataset import DatasetMixin
import imageio
import numpy as np
import tqdm

from detector.hand_dataset.bbox_dataset import HandBBoxDataset as HandDataset
from detector.graphics import camera
from detector.hand_dataset.common_dataset import NUM_KEYPOINTS, STANDARD_KEYPOINT_NAMES, COLOR_MAP, EDGES, ROOT_IDX, REF_EDGE
from detector.hand_dataset.common_dataset import make_keypoint_converter, wrist2palm
from detector.hand_dataset.geometry_utils import DATA_CONVENTION


# auxiliary function
def depth_two_uint8_to_float(top_bits, bottom_bits):
    """
    Converts a RGB-coded depth into float valued depth.
    Taken from sample script of RHD dataset
    """
    depth_map = (top_bits * 2 ** 8 + bottom_bits).astype('float32')
    depth_map /= float(2 ** 16 - 1)
    depth_map *= 5.0
    return depth_map


"""
Keypoints available:
0: left wrist, 1-4: left thumb [tip to palm], 5-8: left index, ..., 17-20: left pinky,
21: right wrist, 22-25: right thumb, ..., 38-41: right pinky
"""

imH = imW = 320
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

ONESIDE_DEFAULT_KEYPOINT_NAMES = []

for k in ["wrist", "thumb", "index", "middle", "ring", "pinky"]:
    if k == "wrist":
        joint_name = "_".join([k])
        ONESIDE_DEFAULT_KEYPOINT_NAMES.append(joint_name)
    else:
        for p in ["tip", "dip", "pip", "mcp"]:
            joint_name = "_".join([k, p])
            ONESIDE_DEFAULT_KEYPOINT_NAMES.append(joint_name)

assert DEFAULT_KEYPOINT_NAMES == ONESIDE_DEFAULT_KEYPOINT_NAMES

NAME_CONVERTER = make_keypoint_converter(
    root="wrist",
    fingers=["thumb", "index", "middle", "ring", "pinky"],
    parts=["mcp", "pip", "dip", "tip"],
)

INDEX_CONVERTER = [DEFAULT_KEYPOINT_NAMES.index(NAME_CONVERTER[k]) for k in STANDARD_KEYPOINT_NAMES]
KEYPOINT_NAMES = STANDARD_KEYPOINT_NAMES


def create_camera_mat():
    # These values are Taken from anno_all["K"]
    fx = fy = 283.1
    u0 = v0 = 160.
    rgb_camera_intr = camera.CameraIntr(**{"u0": u0, "v0": v0, "fx": fx, "fy": fy})
    depth_camera_intr = camera.CameraIntr(**{"u0": u0, "v0": v0, "fx": fx, "fy": fy})
    cameras = {}
    # Both, rgb and depth, have same camera parameter
    cameras["rgb_camera_intr"] = rgb_camera_intr
    cameras["depth_camera_intr"] = depth_camera_intr
    return cameras


CAMERAS = create_camera_mat()
RGB_CAMERA_INTR = CAMERAS["depth_camera_intr"]
DEPTH_CAMERA_INTR = CAMERAS["rgb_camera_intr"]


class RHDBaseDataset(DatasetMixin):
    def __init__(self, dataset_dir, debug=False, mode="train"):
        self.mode = mode
        self.annotations = self.load_annotations(dataset_dir, debug=debug)
        self.rgb_camera = RGB_CAMERA_INTR
        self.depth_camera = DEPTH_CAMERA_INTR
        self.n_joints = NUM_KEYPOINTS
        self.root_idx = ROOT_IDX
        self.ref_edge = REF_EDGE

    def __len__(self):
        return len(self.annotations)

    def read_depth(self, depth_path):
        depth = imageio.imread(depth_path)
        top_bits = depth[:, :, 0]
        bottom_bits = depth[:, :, 1]
        depth = (top_bits * 2 ** 8 + bottom_bits).astype('float32')
        # expand dim so that depth.shape is (C,H,W)
        return np.expand_dims(depth, axis=0)

    def get_example(self, i):
        return self.annotations[i]

    def load_annotations(self, dataset_dir, debug=False):  # load annotations of this set
        logger.info("load annotations mode = {}".format(self.mode))
        logger.info("dataset_dir = {}".format(dataset_dir))
        annotations = []

        if self.mode == "train":
            directory = os.path.join(dataset_dir, "training")
            with open(os.path.join(directory, 'anno_training.pickle'), 'rb') as f:
                anno_all = pickle.load(f)
        else:
            directory = os.path.join(dataset_dir, "evaluation")
            with open(os.path.join(directory, 'anno_evaluation.pickle'), 'rb') as f:
                anno_all = pickle.load(f)
        num_data = len(list(anno_all.keys()))
        for sample_id in tqdm.tqdm(range(num_data)):
            anno = anno_all[sample_id]
            file_format = "{:05d}.png".format(sample_id)
            rgb_path = os.path.join(directory, "color", file_format)
            # mask_path = os.path.join(directory, "mask", file_format)
            depth_path = os.path.join(directory, "depth", file_format)

            if DATA_CONVENTION == "ZYX":
                kp_zyx = anno["xyz"][:, ::-1]
                vu = RGB_CAMERA_INTR.zyx2vu(kp_zyx)
                vs = vu[:, 0]
                us = vu[:, 1]
                joint_left = kp_zyx[:NUM_KEYPOINTS]
                joint_right = kp_zyx[NUM_KEYPOINTS:]
            else:
                kp_xyz = anno["xyz"]
                uv = RGB_CAMERA_INTR.zyx2vu(kp_xyz)
                us = uv[:, 0]
                vs = uv[:, 1]
                joint_left = kp_xyz[:NUM_KEYPOINTS]
                joint_right = kp_xyz[NUM_KEYPOINTS:]

            example = {
                "rgb_path": rgb_path,
                "depth_path": depth_path,
            }
            hand_side = []
            rgb_joint = []
            depth_joint = []
            for hs, joint in zip(["left", "right"], [joint_left, joint_right]):
                if not np.logical_and(np.all(0 < us), np.all(us < imW)):
                    continue
                if not np.logical_and(np.all(0 < vs), np.all(vs < imH)):
                    continue
                joint = joint[INDEX_CONVERTER]
                joint = wrist2palm(joint)
                rgb_joint.append(1000 * joint)
                depth_joint.append(1000 * joint)
                hand_side.append(hs)
            if not hand_side:
                continue
            example["hand_side"] = np.array(hand_side)
            example["rgb_joint"] = np.array(rgb_joint)
            example["depth_joint"] = np.array(depth_joint)
            annotations.append(example)
            if debug and len(annotations) > 1:
                # only read a few sample
                break
        return annotations


def get_base_dataset(dataset_dir, **kwargs):
    base = RHDBaseDataset(dataset_dir, **kwargs)
    return base


def get_rhd_dataset(dataset_dir, param, **kwargs):
    base = get_base_dataset(dataset_dir, **kwargs)
    dataset = HandDataset(base, param)
    return dataset


def get_dataset(dataset_dir, param, **kwargs):
    dataset = get_rhd_dataset(dataset_dir, param, **kwargs)
    return dataset


def visualize_dataset(dataset_dir):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # NOQA
    from chainercv.utils import read_image
    from visualizations import vis_point, vis_edge

    dataset = RHDBaseDataset(dataset_dir, debug=True, mode="train")
    example = dataset.get_example(1)
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
    color = [COLOR_MAP[k] for k in STANDARD_KEYPOINT_NAMES]
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
    dataset_dir = os.path.expanduser("~/dataset/RHD_published_v2")
    visualize_dataset(dataset_dir)
