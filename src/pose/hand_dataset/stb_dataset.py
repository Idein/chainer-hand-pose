import logging

logger = logging.getLogger(__name__)

import os

from chainer.dataset import DatasetMixin
import imageio
import numpy as np
from scipy.io import loadmat

from pose.hand_dataset.dataset import HandPoseDataset as HandDataset
from pose.graphics import camera
from pose.hand_dataset.common_dataset import NUM_KEYPOINTS, STANDARD_KEYPOINT_NAMES, COLOR_MAP, EDGES, ROOT_IDX, REF_EDGE
from pose.hand_dataset.common_dataset import make_keypoint_converter
from pose.hand_dataset.common_dataset import DATA_CONVENTION

_B_COUNTINGS = ["B{}Counting".format(i) for i in [2, 3, 4, 5, 6]]
_B_RANDOMS = ["B{}Random".format(i) for i in [2, 3, 4, 5, 6]]
TRAIN_SEQUENCES = _B_COUNTINGS + _B_RANDOMS

_B_COUNTINGS = ["B{}Counting".format(i) for i in [1]]
_B_RANDOMS = ["B{}Random".format(i) for i in [1]]
VALIDATE_SEQUENCES = _B_COUNTINGS + _B_RANDOMS

DEFAULT_KEYPOINT_NAMES = [
    "palm",
    "little_mcp",
    "little_pip",
    "little_dip",
    "little_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "thumb_mcp",
    "thumb_pip",
    "thumb_dip",
    "thumb_tip",
]

assert len(DEFAULT_KEYPOINT_NAMES) == NUM_KEYPOINTS

NAME_CONVERTER = make_keypoint_converter(
    root="palm",
    fingers=["thumb", "index", "middle", "ring", "little"],
    parts=["mcp", "pip", "dip", "tip"],
    sep="_",
)

INDEX_CONVERTER = [DEFAULT_KEYPOINT_NAMES.index(NAME_CONVERTER[k]) for k in STANDARD_KEYPOINT_NAMES]
KEYPOINT_NAMES = STANDARD_KEYPOINT_NAMES


def create_camera_mat():
    """
    Camera parameter of Intel Real Sense F200 active depth camera.
    These parameters are taken from readme.txt of STB dataset.
    1. Camera parameters
    (1) Point Grey Bumblebee2 stereo camera:
    base line = 120.054
    fx = 822.79041
    fy = 822.79041
    tx = 318.47345
    ty = 250.31296
    (2) Intel Real Sense F200 active depth camera:
    fx color = 607.92271
    fy color = 607.88192
    tx color = 314.78337
    ty color = 236.42484
    fx depth = 475.62768
    fy depth = 474.77709
    tx depth = 336.41179
    ty depth = 238.77962
    rotation vector = [0.00531   -0.01196  0.00301] (use Rodrigues' rotation formula to transform it into rotation matrix)
    translation vector = [-24.0381   -0.4563   -1.2326]
    (rotation and translation vector can transform the coordinates relative to color camera to those relative to depth camera)
    """
    fx_color = 607.92271
    fy_color = 607.88192
    tx_color = 314.78337
    ty_color = 236.42484
    rgb_camera_intr = camera.CameraIntr(
        **{"u0": tx_color, "v0": ty_color, "fx": fx_color, "fy": fy_color}
    )

    fx_depth = 475.62768
    fy_depth = 474.77709
    tx_depth = 336.41179
    ty_depth = 238.77962
    depth_camera_intr = camera.CameraIntr(
        **{"u0": tx_depth, "v0": ty_depth, "fx": fx_depth, "fy": fy_depth}
    )

    def Rodrigues(rotation_vector):
        theta = np.linalg.norm(rotation_vector)
        rv = rotation_vector / theta
        rr = np.array([[rv[i] * rv[j] for j in range(3)] for i in range(3)])
        R = np.cos(theta) * np.eye(3)
        R += (1 - np.cos(theta)) * rr
        R += np.sin(theta) * np.array([
            [0, -rv[2], rv[1]],
            [rv[2], 0, -rv[0]],
            [-rv[1], rv[0], 0],
        ])
        return R

    rotation_vector = np.array([0.00531, -0.01196, 0.00301])
    R = Rodrigues(rotation_vector)
    # transpose to fit our camera class interface
    R = R.transpose()
    # apply minus to fit our camera class interface
    translation_vector = -np.array([-24.0381, - 0.4563, - 1.2326])
    rgb_camera_extr = camera.CameraExtr(R, translation_vector)
    depth_camera_extr = camera.CameraExtr(R, np.zeros(3))
    cameras = {}
    cameras["rgb_camera_intr"] = rgb_camera_intr
    cameras["depth_camera_intr"] = depth_camera_intr
    cameras["rgb_camera_extr"] = rgb_camera_extr
    cameras["depth_camera_extr"] = depth_camera_extr
    return cameras


CAMERAS = create_camera_mat()
RGB_CAMERA_INTR = CAMERAS["rgb_camera_intr"]
DEPTH_CAMERA_INTR = CAMERAS["depth_camera_intr"]
RGB_CAMERA_EXTR = CAMERAS["rgb_camera_extr"]
DEPTH_CAMERA_EXTR = CAMERAS["depth_camera_extr"]


class STBBaseDataset(DatasetMixin):
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

    def get_example(self, i):
        return self.annotations[i]

    def read_depth(self, depth_path):
        depth = imageio.imread(depth_path)
        depth = depth[:, :, 0] + 256 * depth[:, :, 1]
        # expand dim so that depth.shape is (C,H,W)
        return np.expand_dims(depth, axis=0)

    def load_annotations(self, dataset_dir, debug=False):
        logger.info("> load annotations mode = {}".format(self.mode))
        logger.info("> dataset_dir = {}".format(dataset_dir))
        image_dir = os.path.join(dataset_dir, "images")
        label_dir = os.path.join(dataset_dir, "labels")
        annotations = []
        SEQUENCES = TRAIN_SEQUENCES if self.mode == "train" else VALIDATE_SEQUENCES
        for seq in SEQUENCES:
            matSK = loadmat(
                os.path.join(label_dir, "_".join([seq, "SK.mat"])))
            # (xyz,joint_id,frame_idx) -> (frame_idx,joint_id,xyz)
            annotationsSK = matSK["handPara"].transpose(2, 1, 0)

            num_frames, _, _ = annotationsSK.shape
            step = 2 if self.mode == "train" else 10
            for frame_idx in range(0, num_frames, step):
                depth_path = os.path.join(
                    image_dir, seq, "SK_depth_{}.png".format(frame_idx))
                rgb_path = os.path.join(
                    image_dir, seq, "SK_color_{}.png".format(frame_idx))
                if not os.path.exists(rgb_path):
                    continue
                if not os.path.exists(depth_path):
                    continue
                if DATA_CONVENTION == "ZYX":
                    world_joint = annotationsSK[frame_idx]
                    world_joint = world_joint[:, ::-1]
                    rgb_joint = RGB_CAMERA_EXTR.world_zyx2cam_zyx(world_joint)
                    depth_joint = DEPTH_CAMERA_EXTR.world_zyx2cam_zyx(world_joint)
                else:
                    world_joint = annotationsSK[frame_idx]
                    rgb_joint = RGB_CAMERA_EXTR.world_xyz2cam_xyz(world_joint)
                    depth_joint = DEPTH_CAMERA_EXTR.world_xyz2cam_xyz(world_joint)

                rgb_joint = rgb_joint[INDEX_CONVERTER]
                depth_joint = depth_joint[INDEX_CONVERTER]

                example = {}
                example["hand_side"] = "left"
                example["rgb_path"] = rgb_path
                example["rgb_joint"] = rgb_joint
                example["depth_path"] = depth_path
                example["depth_joint"] = depth_joint
                annotations.append(example)

            if debug:
                # only read a few sample
                break
        logger.info("> Done loading annotations")
        return annotations


def get_base_dataset(dataset_dir, param, **kwargs):
    base = STBBaseDataset(dataset_dir, **kwargs)
    return base


def get_stb_dataset(dataset_dir, param, **kwargs):
    base = get_base_dataset(dataset_dir, param, **kwargs)
    dataset = HandDataset(base, param)
    return dataset


def get_dataset(dataset_dir, param, **kwargs):
    dataset = get_stb_dataset(dataset_dir, param, **kwargs)
    return dataset


def visualize_dataset(dataset_dir):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # NOQA
    from chainercv.utils import read_image
    from pose.visualizations import vis_point, vis_edge

    dataset = STBBaseDataset(dataset_dir, debug=True, mode="train")
    example = dataset.get_example(128)
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
    dataset_dir = os.path.expanduser("~/dataset/stb")
    visualize_dataset(dataset_dir)
