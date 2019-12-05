import logging

logger = logging.getLogger(__name__)

import os
import glob
import pickle

from chainer.dataset import DatasetMixin
import numpy as np
import pandas as pd

from detector.hand_dataset.bbox_dataset import HandBBoxDataset as HandDataset
from detector.graphics import camera
from detector.hand_dataset.common_dataset import NUM_KEYPOINTS, STANDARD_KEYPOINT_NAMES, ROOT_IDX, REF_EDGE, COLOR_MAP, EDGES
from detector.hand_dataset.common_dataset import make_keypoint_converter
from detector.hand_dataset.geometry_utils import DATA_CONVENTION, rodrigues

DEFAULT_KEYPOINT_NAMES = [
    'F4_KNU1_A',
    'F4_KNU1_B',
    'F4_KNU2_A',
    'F4_KNU3_A',
    'F3_KNU1_A',
    'F3_KNU1_B',
    'F3_KNU2_A',
    'F3_KNU3_A',
    'F1_KNU1_A',
    'F1_KNU1_B',
    'F1_KNU2_A',
    'F1_KNU3_A',
    'F2_KNU1_A',
    'F2_KNU1_B',
    'F2_KNU2_A',
    'F2_KNU3_A',
    'TH_KNU1_A',
    'TH_KNU1_B',
    'TH_KNU2_A',
    'TH_KNU3_A',
    'PALM_POSITION',
    # 'PALM_NORMAL' # ignore
]

assert len(DEFAULT_KEYPOINT_NAMES) == NUM_KEYPOINTS

NAME_CONVERTER = make_keypoint_converter(
    root="PALM_POSITION",
    fingers=["TH", "F4", "F3", "F2", "F1"],
    parts=["KNU1_B", "KNU1_A", "KNU2_A", "KNU3_A"],
)

INDEX_CONVERTER = [DEFAULT_KEYPOINT_NAMES.index(NAME_CONVERTER[k]) for k in STANDARD_KEYPOINT_NAMES]
KEYPOINT_NAMES = STANDARD_KEYPOINT_NAMES


def create_camera_mat():
    fx = 614.878
    fy = 615.479
    u0 = 313.219
    v0 = 231.288
    rgb_camera_intr = camera.CameraIntr(**{"u0": u0, "v0": v0, "fx": fx, "fy": fy})
    cameras = {"rgb_camera_intr": rgb_camera_intr}
    return cameras


CAMERAS = create_camera_mat()

RGB_CAMERA_INTR = CAMERAS["rgb_camera_intr"]


class MultiViewBaseDataset(DatasetMixin):
    def __init__(self, dataset_dir, debug=False, mode="train"):
        self.rgb_camera = RGB_CAMERA_INTR
        self.depth_camera = None
        self.n_joints = NUM_KEYPOINTS
        self.root_idx = ROOT_IDX
        self.ref_edge = REF_EDGE
        self.mode = mode
        self.dataset_dir = dataset_dir
        self.annotations = self.load_annotations(dataset_dir, debug=debug)

    def __len__(self):
        return len(self.annotations)

    def get_example(self, i):
        return self.annotations[i]

    def lemma(self, use_aug_sample, data_folder, cam_idx, num_frames, debug):
        annotations = []
        calib_dir = os.path.join(self.dataset_dir, "calibrations", data_folder,
                                 "webcam_{}".format(cam_idx))
        with open(os.path.join(calib_dir, "rvec.pkl"), 'rb') as f:
            calibR = pickle.load(f, encoding='latin1')
        with open(os.path.join(calib_dir, "tvec.pkl"), 'rb') as f:
            calibT = pickle.load(f, encoding='latin1')
        for frame_idx in range(num_frames):
            if use_aug_sample:
                rgb_path = os.path.join(self.dataset_dir, "augmented_samples", data_folder,
                                        "{}_webcam_{}.jpg".format(frame_idx, cam_idx))
            else:
                rgb_path = os.path.join(self.dataset_dir, "annotated_frames", data_folder,
                                        "{}_webcam_{}.jpg".format(frame_idx, cam_idx))
            joint_path = os.path.join(self.dataset_dir, "annotated_frames", data_folder,
                                      "{}_joints.txt".format(frame_idx))
            df = pd.read_csv(joint_path, sep=' ', usecols=[1, 2, 3], header=None)
            rgb_joint = df.to_numpy()
            rgb_joint = rgb_joint[:21]  # ignore last keypoint i.e. PALM_NORMAL

            R = rodrigues(calibR.squeeze())
            rgb_joint = rgb_joint.dot(R.transpose()) + calibT.transpose()
            if not os.path.exists(rgb_path):
                continue
            if DATA_CONVENTION == "ZYX":
                # xyz -> zyx
                rgb_joint = rgb_joint[:, ::-1]
            # sort keypoint to our standard order
            rgb_joint = rgb_joint[INDEX_CONVERTER]
            example = {"hand_side": "right", "rgb_path": rgb_path, "rgb_joint": rgb_joint}
            annotations.append(example)
            if debug and len(annotations) > 10:
                return annotations
        return annotations

    def load_annotations(self, dataset_dir, debug=False):
        logger.info("> load multiview")
        annotated_dirs = sorted(os.listdir(os.path.join(dataset_dir, "annotated_frames")))
        queries = []
        for use_aug_sample in [False, True]:
            for data_idx in range(len(annotated_dirs)):
                data_folder = annotated_dirs[data_idx]
                if self.mode == "train":
                    if data_folder in ["data_1", "data_10", "data_20"]:
                        continue
                else:
                    if not data_folder in ["data_1", "data_10", "data_20"]:
                        continue
                pattern = os.path.join(dataset_dir, "annotated_frames", data_folder, "*_joints.txt")
                num_frames = len(glob.glob(pattern))
                for cam_idx in [1, 2, 3, 4]:
                    queries.append([use_aug_sample, data_folder, cam_idx, num_frames, debug])
        import multiprocessing
        logger.info("> use multiprocessing")
        with multiprocessing.Pool() as pool:
            anns = pool.starmap(self.lemma, queries)
        logger.info("> finish !")
        annotations = []
        for a in anns:
            annotations += a
        return annotations


def get_base_dataset(dataset_dir, **kwargs):
    base = MultiViewBaseDataset(dataset_dir, **kwargs)
    return base


def get_multiview_dataset(dataset_dir, param, **kwargs):
    base = get_base_dataset(dataset_dir, **kwargs)
    dataset = HandDataset(base, param)
    return dataset


def get_dataset(dataset_dir, param, **kwargs):
    dataset = get_multiview_dataset(dataset_dir, param, **kwargs)
    return dataset


def visualize_dataset(dataset_dir):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # NOQA
    from chainercv.utils import read_image
    from visualizations import vis_point, vis_edge

    dataset = MultiViewBaseDataset(dataset_dir, debug=False)
    print(len(dataset))
    i = np.random.choice(len(dataset))
    example = dataset.get_example(i)
    rgb_path = example["rgb_path"]
    rgb_joint = example["rgb_joint"]
    print(rgb_path)
    img = read_image(rgb_path)
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(211)
    ax3 = fig.add_subplot(212, projection="3d")
    rgb_vu = dataset.rgb_camera.zyx2vu(rgb_joint)
    color = [COLOR_MAP[k] for k in STANDARD_KEYPOINT_NAMES]
    edge_color = [COLOR_MAP[s, t] for s, t in EDGES]
    vis_point(rgb_vu, img=img, color=color, ax=ax1)
    vis_edge(rgb_vu, indices=EDGES, color=edge_color, ax=ax1)

    vis_point(rgb_joint, color=color, ax=ax3)
    vis_edge(rgb_joint, indices=EDGES, color=edge_color, ax=ax3)

    for ax in [ax3]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(-65, -90)

    plt.show()


if __name__ == '__main__':
    dataset_dir = os.path.expanduser("~/dataset/multiview_hand")
    visualize_dataset(dataset_dir)
