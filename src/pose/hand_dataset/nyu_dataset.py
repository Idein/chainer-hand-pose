import os
import logging

logger = logging.getLogger(__name__)

from chainer.dataset import DatasetMixin
import imageio
import numpy as np
from scipy.io import loadmat
import tqdm

from pose.graphics import camera
from pose.hand_dataset.dataset import HandPoseDataset
from pose.hand_dataset.common_dataset import NUM_KEYPOINTS, STANDARD_KEYPOINT_NAMES, COLOR_MAP, EDGES, ROOT_IDX, REF_EDGE
from pose.hand_dataset.common_dataset import make_keypoint_converter
from pose.hand_dataset.common_dataset import DATA_CONVENTION

ORIGINAL_NUM_KEYPOINTS = 36
ORIGINAL_KEYPOINT_NAMES = [
    "F1_KNU3_A",
    "F1_KNU3_B",
    "F1_KNU2_A",
    "F1_KNU2_B",
    "F1_KNU1_A",
    "F1_KNU1_B",
    "F2_KNU3_A",
    "F2_KNU3_B",
    "F2_KNU2_A",
    "F2_KNU2_B",
    "F2_KNU1_A",
    "F2_KNU1_B",
    "F3_KNU3_A",
    "F3_KNU3_B",
    "F3_KNU2_A",
    "F3_KNU2_B",
    "F3_KNU1_A",
    "F3_KNU1_B",
    "F4_KNU3_A",
    "F4_KNU3_B",
    "F4_KNU2_A",
    "F4_KNU2_B",
    "F4_KNU1_A",
    "F4_KNU1_B",
    "TH_KNU3_A",
    "TH_KNU3_B",
    "TH_KNU2_A",
    "TH_KNU2_B",
    "TH_KNU1_A",
    "TH_KNU1_B",
    "PALM_1",
    "PALM_2",
    "PALM_3",
    "PALM_4",
    "PALM_5",
    "PALM_6",
]

DEFAULT_KEYPOINT_NAMES = [
    "little_tip",
    "little_dip",
    "little_pip",
    "little_mcp",
    "ring_tip",
    "ring_dip",
    "ring_pip",
    "ring_mcp",
    "middle_tip",
    "middle_dip",
    "middle_pip",
    "middle_mcp",
    "index_tip",
    "index_dip",
    "index_pip",
    "index_mcp",
    "thumb_tip",
    "thumb_dip",
    "thumb_pip",
    "thumb_mcp",
    "root",
]

TARGET_CONVERTER = {
    "root": "PALM_3",  # palm
    # "root": "PALM_6",  # so called wrist
}

for f, fID in zip(["thumb", "index", "middle", "ring", "little"], ["TH", "F4", "F3", "F2", "F1"]):
    for p, pID in zip(["tip", "dip", "pip", "mcp"], ["KNU3_A", "KNU3_B", "KNU2_B", "KNU1_A"]):
        key = "_".join([f, p])
        value = "_".join([fID, pID])
        TARGET_CONVERTER[key] = value

TARGET_INDICES = [ORIGINAL_KEYPOINT_NAMES.index(TARGET_CONVERTER[k]) for k in DEFAULT_KEYPOINT_NAMES]

assert len(ORIGINAL_KEYPOINT_NAMES) == ORIGINAL_NUM_KEYPOINTS
assert len(DEFAULT_KEYPOINT_NAMES) == NUM_KEYPOINTS

NAME_CONVERTER = make_keypoint_converter(
    root="root",
    fingers=["thumb", "index", "middle", "ring", "little"],
    parts=["mcp", "pip", "dip", "tip"],
)

INDEX_CONVERTER = [DEFAULT_KEYPOINT_NAMES.index(NAME_CONVERTER[k]) for k in STANDARD_KEYPOINT_NAMES]
KEYPOINT_NAMES = STANDARD_KEYPOINT_NAMES


def create_camera_mat():
    # These parameters are taken from convert_xyz_to_uvd.m provided by NYU dataset.
    u0 = 640 / 2
    v0 = 480 / 2
    fx = 588.036865
    fy = 587.075073

    rgb_camera_intr = camera.CameraIntr(**{"u0": u0, "v0": v0, "fx": fx, "fy": fy})
    depth_camera_intr = camera.CameraIntr(**{"u0": u0, "v0": v0, "fx": fx, "fy": fy})

    cameras = {}
    cameras["rgb_camera_intr"] = rgb_camera_intr
    cameras["depth_camera_intr"] = depth_camera_intr
    return cameras


CAMERAS = create_camera_mat()
RGB_CAMERA_INTR = CAMERAS["rgb_camera_intr"]
DEPTH_CAMERA_INTR = CAMERAS["depth_camera_intr"]


class NYUBaseDataset(DatasetMixin):
    def __init__(self, dataset_dir, debug=False, mode="train", use_synth=False):
        self.rgb_camera = RGB_CAMERA_INTR
        self.depth_camera = DEPTH_CAMERA_INTR
        self.n_joints = NUM_KEYPOINTS
        self.root_idx = ROOT_IDX
        self.ref_edge = REF_EDGE
        self.mode = mode
        self.use_synth = use_synth
        self.annotations = self.load_annotations(dataset_dir, debug=debug)

    def __len__(self):
        return len(self.annotations)

    def read_depth(self, depth_path):
        depth = imageio.imread(depth_path)
        topbit = depth[:, :, 1]
        lowerbit = depth[:, :, 2]
        depth = 2 ** 8 * topbit + lowerbit
        depth = np.expand_dims(depth, axis=0)
        return depth

    def get_example(self, i):
        return self.annotations[i]

    def load_annotations(self, dataset_dir, debug=False):
        logger.info("load annotations mode {}".format(self.mode))
        logger.info("dataset_dir = {}".format(dataset_dir))
        annotations = []
        if self.mode == "train":
            fmt = os.path.join(dataset_dir, "dataset", "train", "{0}_{1}_{2:07d}.png")
            matfile = os.path.join(dataset_dir, "dataset", "train", "joint_data.mat")
        else:
            fmt = os.path.join(dataset_dir, "dataset", "test", "{0}_{1}_{2:07d}.png")
            matfile = os.path.join(dataset_dir, "dataset", "test", "joint_data.mat")
        mat = loadmat(matfile, squeeze_me=True)
        for k in [1, 2, 3]:
            num_frames = mat["joint_xyz"][k - 1].shape[0]
            step = 5 if self.mode == "train" else 10
            for f in tqdm.tqdm(range(1, num_frames + 1, step)):
                rgb_path = fmt.format("rgb", k, f)
                depth_path = fmt.format("depth", k, f)
                synthdepth_path = fmt.format("synthdepth", k, f)
                joint = mat["joint_xyz"][k - 1][f - 1].copy()
                # flip v-direction for our purpose
                joint[:, 1] = -joint[:, 1]
                # restrict 36 pts -> 21 pts
                joint = joint[TARGET_INDICES]
                if DATA_CONVENTION == "ZYX":
                    joint = joint[:, ::-1]

                joint = joint[INDEX_CONVERTER]
                example = {}
                example["rgb_path"] = rgb_path
                if self.use_synth:
                    example["depth_path"] = synthdepth_path
                else:
                    example["depth_path"] = depth_path
                example["rgb_joint"] = joint.copy()
                example["depth_joint"] = joint.copy()
                annotations.append(example)
                if debug:
                    break
        return annotations


def get_nyu_dataset(dataset_dir, param, **kwargs):
    base = NYUBaseDataset(dataset_dir, **kwargs)
    dataset = HandPoseDataset(base, param)
    return dataset


def visualize_dataset(dataset_dir):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # NOQA
    from chainercv.utils import read_image
    from pose.visualizations import vis_point, vis_edge

    dataset = NYUBaseDataset(dataset_dir, debug=True, mode="train")
    example = dataset.get_example(0)
    camera_joint = example["rgb_joint"]
    depth_joint = example["depth_joint"]
    rgb_path = example["rgb_path"]
    depth_path = example["depth_path"]
    logger.info("> read {}".format(rgb_path))
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
    dataset_dir = os.path.expanduser("~/dataset/nyu_hand_dataset_v2")
    visualize_dataset(dataset_dir)
