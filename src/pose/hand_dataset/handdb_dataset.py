import os
import glob
import json
import numpy as np
import logging

logger = logging.getLogger(__name__)

from chainer.dataset import DatasetMixin

from pose.hand_dataset.dataset import HandPoseDataset as HandDataset
from pose.graphics import camera
from pose.hand_dataset.common_dataset import NUM_KEYPOINTS, STANDARD_KEYPOINT_NAMES, COLOR_MAP, EDGES, ROOT_IDX, REF_EDGE
from pose.hand_dataset.common_dataset import make_keypoint_converter, wrist2palm
from pose.hand_dataset.common_dataset import DATA_CONVENTION

# Puseudo image center and focal length
# This dataset provides only 2D image. To utilize our script that use 3D information,
# we will create puseudo 3D coordinate and camera matrix.

U0 = V0 = 300  # Puseudo image center
F = 500  # Puseudo Focal length

RGB_CAMERA_INTR = camera.CameraIntr(fx=F, fy=F, u0=U0, v0=V0)
KEYPOINT_NAMES = STANDARD_KEYPOINT_NAMES


class HandDBBaseDataset(DatasetMixin):
    def __init__(self, dataset_dir, mode="train", debug=False):
        self.mode = mode
        self.rgb_camera = RGB_CAMERA_INTR
        self.depth_camera = None
        self.n_joints = NUM_KEYPOINTS
        self.root_idx = ROOT_IDX
        self.ref_edge = REF_EDGE
        self.annotations = self.load_annotations(dataset_dir, debug=debug)

    def __len__(self):
        return len(self.annotations)

    def get_example(self, i):
        return self.annotations[i]

    def load_annotations(self, dataset_dir, debug=False):
        # TODO: utilize hand side
        logger.info("load annotations mode = {}".format(self.mode))
        logger.info("dataset_dir = {}".format(dataset_dir))
        annotations = []
        manual_annotations_dir = os.path.join(dataset_dir, "hand_labels")
        if self.mode == "train":
            manual_dir = os.path.join(manual_annotations_dir, "manual_train")
        else:
            manual_dir = os.path.join(manual_annotations_dir, "manual_test")
        json_files = sorted(glob.glob(os.path.join(manual_dir, "*.json")))
        for file in json_files:
            with open(file, 'r') as f:
                anno = json.load(f)
                rgb_joint = np.array(anno["hand_pts"])
                rgb_joint = rgb_joint[:, :2]
                is_left = anno["is_left"]
            rgb_path = os.path.splitext(file)[0] + ".jpg"
            if DATA_CONVENTION == "ZYX":
                # xy -> yx
                rgb_joint = rgb_joint[:, ::-1] - np.array([[U0, V0]])
                zyx = F * np.ones((NUM_KEYPOINTS, 3))
                zyx[:, 1:] = rgb_joint
                rgb_joint = zyx
            else:
                xyz = F * np.ones((NUM_KEYPOINTS, 3))
                xyz[:, :2] = rgb_joint
                rgb_joint = xyz
            rgb_joint = wrist2palm(rgb_joint)
            example = {}
            hand_side = "left" if int(is_left) != 0 else "right"
            example["hand_side"] = hand_side
            example["rgb_path"] = rgb_path
            example["rgb_joint"] = rgb_joint
            annotations.append(example)
            if debug and len(annotations) > 10:
                break

        synth_annotations_dir = os.path.join(dataset_dir, "hand_labels_synth")
        synths = [2, 3] if self.mode == "train" else [4]
        for synth in synths:
            pattern = os.path.join(synth_annotations_dir, "synth{}".format(synth), "*.json")
            json_files = sorted(glob.glob(pattern))
            for file in json_files:
                with open(file, 'r') as f:
                    anns = json.load(f)
                    rgb_joint = np.array(anns["hand_pts"])
                    rgb_joint = rgb_joint[:, :2]
                    is_left = anns["is_left"]
                rgb_path = file.replace("json", "jpg")
                if DATA_CONVENTION == "ZYX":
                    # xy -> yx
                    rgb_joint = rgb_joint[:, ::-1] - np.array([[U0, V0]])
                    zyx = F * np.ones((NUM_KEYPOINTS, 3))
                    zyx[:, 1:] = rgb_joint
                    rgb_joint = zyx
                else:
                    xyz = F * np.ones((NUM_KEYPOINTS, 3))
                    xyz[:, :2] = rgb_joint
                    rgb_joint = xyz
                rgb_joint = wrist2palm(rgb_joint)
                example = {}
                hand_side = "left" if int(is_left) != 0 else "right"
                example["hand_side"] = hand_side
                example["rgb_path"] = rgb_path
                example["rgb_joint"] = rgb_joint
                annotations.append(example)
                if debug and len(annotations) > 10:
                    break

        if self.mode == "train":
            panoptic = os.path.join(dataset_dir, "hand143_panopticdb")
            # exclude synth1 cuz these files under synth1 has no 21 points
            json_file = os.path.join(panoptic, "hands_v143_14817.json")
            with open(json_file, 'r') as f:
                anns = json.load(f)
            for idx in range(len(anns["root"])):
                anno = anns["root"][idx]
                rgb_path = os.path.join(panoptic, anno["img_paths"])
                rgb_joint = np.array(anno["joint_self"])
                rgb_joint = rgb_joint[:, :2]
                if DATA_CONVENTION == "ZYX":
                    # xy -> yx
                    rgb_joint = rgb_joint[:, ::-1] - np.array([[U0, V0]])
                    zyx = F * np.ones((NUM_KEYPOINTS, 3))
                    zyx[:, 1:] = rgb_joint
                    rgb_joint = zyx
                else:
                    xyz = F * np.ones((NUM_KEYPOINTS, 3))
                    xyz[:, :2] = rgb_joint
                    rgb_joint = xyz
                rgb_joint = wrist2palm(rgb_joint)
                example = {}
                example["hand_side"] = "right"
                example["rgb_path"] = rgb_path
                example["rgb_joint"] = rgb_joint
                annotations.append(example)
                if debug and len(annotations) > 10:
                    break

        return annotations


def get_base_dataset(dataset_dir, **kwargs):
    base = HandDBBaseDataset(dataset_dir, **kwargs)
    return base


def get_handdb_dataset(dataset_dir, param, **kwargs):
    base = get_base_dataset(dataset_dir, **kwargs)
    dataset = HandDataset(base, param)
    return dataset


def get_dataset(dataset_dir, param, **kwargs):
    dataset = get_handdb_dataset(dataset_dir, param, **kwargs)
    return dataset


def visualize_dataset(dataset_dir):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # NOQA
    from chainercv.utils import read_image
    from pose.visualizations import vis_point, vis_edge

    dataset = HandDBBaseDataset(dataset_dir, debug=True, mode="train")
    example = dataset.get_example(0)
    rgb_joint = example["rgb_joint"]
    rgb_path = example["rgb_path"]
    img = read_image(rgb_path)
    fig = plt.figure(figsize=(5, 10))
    ax1 = fig.add_subplot(211)
    ax3 = fig.add_subplot(212, projection="3d")
    rgb_vu = RGB_CAMERA_INTR.zyx2vu(rgb_joint)
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
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    dataset_dir = os.path.expanduser("~/dataset/handdb_dataset")
    visualize_dataset(dataset_dir)
