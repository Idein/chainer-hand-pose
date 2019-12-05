import logging

logger = logging.getLogger(__name__)

import os

from chainer.dataset import DatasetMixin

import numpy as np
import tqdm

from pose.graphics import camera
from pose.hand_dataset.dataset import HandPoseDataset
from pose.hand_dataset.common_dataset import NUM_KEYPOINTS, STANDARD_KEYPOINT_NAMES, COLOR_MAP, EDGES, ROOT_IDX, REF_EDGE
from pose.hand_dataset.common_dataset import make_keypoint_converter, wrist2palm
from pose.hand_dataset.common_dataset import DATA_CONVENTION

SUBJECTS = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
GESTURES = ["1", "2", "3", "4", "5", "6", "7", "8",
            "9", "I", "IP", "L", "MP", "RP", "T", "TIP", "Y"]

DEFAULT_KEYPOINT_NAMES = [
    "wrist",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "little_mcp",
    "little_pip",
    "little_dip",
    "little_tip",
    "thumb_mcp",
    "thumb_pip",
    "thumb_dip",
    "thumb_tip",
]

assert len(DEFAULT_KEYPOINT_NAMES) == NUM_KEYPOINTS

NAME_CONVERTER = make_keypoint_converter(
    root="wrist",
    fingers=["thumb", "index", "middle", "ring", "little"],
    parts=["mcp", "pip", "dip", "tip"],
)

INDEX_CONVERTER = [DEFAULT_KEYPOINT_NAMES.index(NAME_CONVERTER[k]) for k in STANDARD_KEYPOINT_NAMES]
KEYPOINT_NAMES = STANDARD_KEYPOINT_NAMES


def create_camera_mat():
    """
    The camera intrinsic parameters are: principle point = image center(160, 120), focal length = 241.42.
    """
    fx = fy = 241.42
    u0, v0 = 160, 120
    depth_camera_intr = camera.CameraIntr(**{"u0": u0, "v0": v0, "fx": fx, "fy": fy})
    cameras = {}
    cameras["depth_camera_intr"] = depth_camera_intr
    return cameras


CAMERAS = create_camera_mat()
DEPTH_CAMERA_INTR = CAMERAS["depth_camera_intr"]


def load_joint_file(joint_file):
    anns = np.loadtxt(
        joint_file,
        skiprows=1  # ignore first row
    )
    return anns


def load_table(binary_file):
    fsize = os.path.getsize(binary_file)
    sizeof_I = 4
    sizeof_f = 4
    count = (fsize - 6 * sizeof_I) // sizeof_f
    record_dtype = np.dtype(
        [
            ('imginfo', '6I'),
            ('depth_array', '{}f'.format(count))
        ]
    )
    table = np.fromfile(binary_file, dtype=record_dtype)
    return table


class MSRA15BaseDataset(DatasetMixin):
    def __init__(self, dataset_dir, debug=False, mode="train"):
        self.rgb_camera = None
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
        table = load_table(depth_path)
        img_width, img_height, left, top, right, bottom = table["imginfo"].ravel()
        depth_map = table["depth_array"].reshape((bottom - top, right - left))
        img = np.zeros((img_height, img_width), dtype=np.float32)
        img[top:bottom, left:right] = depth_map
        img = np.expand_dims(img, axis=0)
        return img

    def load_annotations(self, dataset_dir, debug=False):
        logger.info("> load annotations mode = {}".format(self.mode))
        logger.info("> dataset_dir = {}".format(dataset_dir))
        annotations = []
        for subject in SUBJECTS:
            for gesture in GESTURES:
                data_dir = os.path.join(dataset_dir, subject, gesture)
                joint_file = os.path.join(data_dir, "joint.txt")
                anns = load_joint_file(joint_file)
                num_frames = anns.shape[0]
                for frame_idx in tqdm.tqdm(range(num_frames)):
                    depth_path = os.path.join(
                        data_dir,
                        "_".join([
                            "{:06d}".format(frame_idx),
                            "depth.bin"
                        ])
                    )
                    joints = anns[frame_idx].reshape(-1, 3)
                    """
                    Transform joint direction in advance to fit our purpose
                    """
                    joints[:, 1] = -joints[:, 1]
                    joints[:, 2] = -joints[:, 2]
                    if DATA_CONVENTION == "ZYX":
                        depth_joint = joints[:, ::-1]
                    else:
                        depth_joint = anns[frame_idx].reshape(-1, 3)

                    depth_joint = depth_joint[INDEX_CONVERTER]
                    depth_joint = wrist2palm(depth_joint)

                    example = {}
                    example["depth_joint"] = depth_joint
                    example["depth_path"] = depth_path
                    annotations.append(example)
            if debug:
                break
        logger.info("> Done load annotations")
        return annotations


def get_msra15_dataset(dataset_dir, param, **kwargs):
    base = MSRA15BaseDataset(dataset_dir, **kwargs)
    dataset = HandPoseDataset(base, param)
    return dataset


def visualize_dataset(dataset_dir):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # NOQA
    from pose.visualizations import vis_point, vis_edge

    dataset = MSRA15BaseDataset(dataset_dir, debug=True, mode="train")
    example = dataset.get_example(128)
    depth_joint = example["depth_joint"]
    depth_path = example["depth_path"]
    depth = dataset.read_depth(depth_path)
    fig = plt.figure(figsize=(5, 10))
    ax2 = fig.add_subplot(211)
    ax4 = fig.add_subplot(212, projection="3d")
    color = [COLOR_MAP[k] for k in KEYPOINT_NAMES]
    edge_color = [COLOR_MAP[s, t] for s, t in EDGES]
    depth_vu = DEPTH_CAMERA_INTR.zyx2vu(depth_joint)

    vis_point(depth_vu, img=depth, color=color, ax=ax2)
    vis_edge(depth_vu, indices=EDGES, color=edge_color, ax=ax2)

    vis_point(depth_joint, color=color, ax=ax4)
    vis_edge(depth_joint, indices=EDGES, color=edge_color, ax=ax4)

    for ax in [ax4]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(-65, -90)

    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    dataset_dir = os.path.expanduser("~/dataset/cvpr15_MSRAHandGestureDB")
    visualize_dataset(dataset_dir)
