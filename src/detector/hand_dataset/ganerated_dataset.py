import logging

logger = logging.getLogger(__name__)

import multiprocessing
import os

from chainer.dataset import DatasetMixin
import chainercv
import numpy as np

from detector.hand_dataset.bbox_dataset import HandBBoxDataset as HandDataset
from detector.graphics import camera
from detector.hand_dataset.common_dataset import NUM_KEYPOINTS, STANDARD_KEYPOINT_NAMES, COLOR_MAP, EDGES, ROOT_IDX, REF_EDGE
from detector.hand_dataset.common_dataset import make_keypoint_converter, wrist2palm
from detector.hand_dataset.geometry_utils import DATA_CONVENTION

MAX_FRAME_IDX = 1024
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
    # See README.txt
    camera_intr = np.array([
        [617.173, 0, 315.453],
        [0, 617.173, 242.259],
        [0, 0, 1],
    ])

    fx = fy = 617.173
    u0 = 315.453
    v0 = 242.259

    rgb_camera_intr = camera.CameraIntr(**{"u0": u0, "v0": v0, "fx": fx, "fy": fy})
    cameras = {}
    cameras["rgb_camera_intr"] = rgb_camera_intr
    return cameras


CAMERAS = create_camera_mat()
RGB_CAMERA_INTR = CAMERAS["rgb_camera_intr"]
DEPTH_CAMERA_INTR = None


class GANeratedBaseDataset(DatasetMixin):

    def __init__(self, dataset_dir, debug=False, mode="train"):
        self.rgb_camera = RGB_CAMERA_INTR
        self.depth_camera = DEPTH_CAMERA_INTR
        self.n_joints = NUM_KEYPOINTS
        self.root_idx = ROOT_IDX
        self.ref_edge = REF_EDGE
        self.mode = mode
        self.dataset_dir = dataset_dir
        self.annotations = self.load_annotations(dataset_dir, debug=debug)

    def __len__(self):
        return len(self.annotations)

    def read_rgb(self, rgb_path):
        return chainercv.utils.read_image(rgb_path)

    def get_example(self, i):
        return self.annotations[i]

    def lemma(self, obj, partition, debug):
        annotations = []
        logger.info("> {} {}".format(obj, partition))
        for frame_idx in range(1, MAX_FRAME_IDX + 1):
            fmt = "{0:04d}_{1}"
            img_dir = os.path.join(self.dataset_dir, "data", obj, partition)
            name = fmt.format(frame_idx, "color_composed.png")
            color_composed_path = os.path.join(img_dir, name)
            name = fmt.format(frame_idx, "crop_params.txt")
            crop_params_path = os.path.join(img_dir, name)
            name = fmt.format(frame_idx, "joint2D.txt")
            joint2D_path = os.path.join(img_dir, name)
            name = fmt.format(frame_idx, "joint_pos.txt")
            joint_pos_path = os.path.join(img_dir, name)
            name = fmt.format(frame_idx, "joint_pos_global.txt")
            joint_pos_global_path = os.path.join(img_dir, name)
            if not os.path.exists(joint_pos_global_path):
                continue
            if DATA_CONVENTION == "ZYX":
                joint = np.loadtxt(joint_pos_global_path, delimiter=',').reshape(-1, 3)
                rgb_joint = joint[:, ::-1]
            else:
                rgb_joint = np.loadtxt(joint_pos_global_path, delimiter=',').reshape(-1, 3)

            rgb_joint = rgb_joint[INDEX_CONVERTER]
            rgb_joint = wrist2palm(rgb_joint)
            crop_u, crop_v, scale = crop_param = np.loadtxt(crop_params_path, delimiter=',')
            rgb_camera = RGB_CAMERA_INTR.translate_camera(y_offset=-crop_v, x_offset=-crop_u)
            rgb_camera = rgb_camera.scale_camera(y_scale=scale, x_scale=scale)
            example = {}
            example["hand_side"] = "left"
            example["rgb_path"] = color_composed_path
            example["rgb_joint"] = rgb_joint
            example["rgb_camera"] = rgb_camera
            annotations.append(example)
            if debug and len(annotations) > 10:
                return annotations
        return annotations

    def load_annotations(self, dataset_dir, debug=False):
        logger.info("> load annotation mode = {}".format(self.mode))
        logger.info("> dataset_dir = {}".format(dataset_dir))
        annotations = []
        queries = []
        for obj in ["noObject", "withObject"]:
            partitions = sorted(os.listdir(os.path.join(dataset_dir, "data", obj)))
            for partition in partitions:
                if self.mode == "train":
                    if os.path.basename(partition) in ["0001", "0002", "0003"]:
                        continue
                else:
                    if not os.path.basename(partition) in ["0001", "0002", "0003"]:
                        continue

                queries.append([obj, partition, debug])

        with multiprocessing.Pool() as pool:
            anns = pool.starmap(self.lemma, queries)
        for a in anns:
            annotations += a
        return annotations


def get_base_dataset(dataset_dir, **kwargs):
    base = GANeratedBaseDataset(dataset_dir, **kwargs)
    return base


def get_ganerated_dataset(dataset_dir, param, **kwargs):
    base = get_base_dataset(dataset_dir, **kwargs)
    dataset = HandDataset(base, param)
    return dataset


def get_dataset(dataset_dir, param, **kwargs):
    dataset = get_ganerated_dataset(dataset_dir, param, **kwargs)
    return dataset


def visualize_dataset(dataset_dir):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # NOQA
    from chainercv.utils import read_image
    from visualizations import vis_point, vis_edge

    dataset = GANeratedBaseDataset(dataset_dir, debug=False, mode="train")
    idx = np.random.choice(len(dataset))
    print(idx, len(dataset))
    example = dataset.get_example(idx)
    rgb_joint = example["rgb_joint"]
    rgb_path = example["rgb_path"]
    rgb = read_image(rgb_path)
    fig = plt.figure(figsize=(5, 10))
    ax2 = fig.add_subplot(211)
    ax4 = fig.add_subplot(212, projection="3d")
    color = [COLOR_MAP[k] for k in KEYPOINT_NAMES]
    edge_color = [COLOR_MAP[s, t] for s, t in EDGES]
    rgb_vu = example["rgb_camera"].zyx2vu(rgb_joint)
    vis_point(rgb_vu, img=rgb, color=color, ax=ax2)
    vis_edge(rgb_vu, indices=EDGES, color=edge_color, ax=ax2)

    vis_point(rgb_joint, color=color, ax=ax4)
    vis_edge(rgb_joint, indices=EDGES, color=edge_color, ax=ax4)

    for ax in [ax4]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(-65, -90)

    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    dataset_dir = os.path.expanduser("~/dataset/GANeratedHands_Release")
    visualize_dataset(dataset_dir)
