import itertools
import logging

logger = logging.getLogger(__name__)

import os

from chainer.dataset import DatasetMixin
import imageio
import numpy as np

from detector.hand_dataset.bbox_dataset import HandBBoxDataset as HandDataset
from detector.graphics import camera
from detector.hand_dataset.common_dataset import NUM_KEYPOINTS, STANDARD_KEYPOINT_NAMES, COLOR_MAP, EDGES, ROOT_IDX, REF_EDGE
from detector.hand_dataset.common_dataset import make_keypoint_converter, wrist2palm
from detector.hand_dataset.geometry_utils import DATA_CONVENTION

ACTION_LIST = [
    'charge_cell_phone',
    'clean_glasses',
    'close_juice_bottle',
    'close_liquid_soap',
    'close_milk',
    'close_peanut_butter',
    'drink_mug',
    'flip_pages',
    'flip_sponge',
    'give_card',
    'give_coin',
    'handshake',
    'high_five',
    'light_candle',
    'open_juice_bottle',
    'open_letter',
    'open_liquid_soap',
    'open_milk',
    'open_peanut_butter',
    'open_soda_can',
    'open_wallet',
    'pour_juice_bottle',
    'pour_liquid_soap',
    'pour_milk',
    'pour_wine',
    'prick',
    'put_salt',
    'put_sugar',
    'put_tea_bag',
    'read_letter',
    'receive_coin',
    'scoop_spoon',
    'scratch_sponge',
    'sprinkle',
    'squeeze_paper',
    'squeeze_sponge',
    'stir',
    'take_letter_from_enveloppe',
    'tear_paper',
    'toast_wine',
    'unfold_glasses',
    'use_calculator',
    'use_flash',
    'wash_sponge',
    'write'
]

SUBJECT_INDICES = [1, 2, 3, 4, 5, 6]
SEQ_INDICES = [1, 2, 3, 4]

# T, I, M, R, P denote Thumb, Index, Middle, Ring, Pinky fingers
DEFAULT_KEYPOINT_NAMES = [
    "Wrist",
    "TMCP",
    "IMCP",
    "MMCP",
    "RMCP",
    "PMCP",
    "TPIP",
    "TDIP",
    "TTIP",
    "IPIP",
    "IDIP",
    "ITIP",
    "MPIP",
    "MDIP",
    "MTIP",
    "RPIP",
    "RDIP",
    "RTIP",
    "PPIP",
    "PDIP",
    "PTIP"
]

COLUMN_NAMES = ["FRAME_ID"]
for kname in DEFAULT_KEYPOINT_NAMES:
    for c in ["X", "Y", "Z"]:
        cname = "_".join([c, kname])
        COLUMN_NAMES.append(cname)

assert len(DEFAULT_KEYPOINT_NAMES) == NUM_KEYPOINTS
NAME_CONVERTER = make_keypoint_converter(
    root="Wrist",
    fingers=["T", "I", "M", "R", "P"],
    parts=["MCP", "PIP", "DIP", "TIP"],
    sep="",
)

INDEX_CONVERTER = [DEFAULT_KEYPOINT_NAMES.index(NAME_CONVERTER[k]) for k in STANDARD_KEYPOINT_NAMES]
KEYPOINT_NAMES = STANDARD_KEYPOINT_NAMES


def create_camera_matrix():
    """
    All parameters are taken from https://github.com/guiggh/hand_pose_action
    """

    # camera parameter for image sensor
    # Image center
    u0 = 935.732544
    v0 = 540.681030
    # Focal Length
    fx = 1395.749023
    fy = 1395.749268

    rgb_camera_intr = camera.CameraIntr(**{"u0": u0, "v0": v0, "fx": fx, "fy": fy})

    # camera parameter for depth sensor
    # Image center:
    depth_u0 = 315.944855
    depth_v0 = 245.287079
    # Focal Length:
    depth_fx = 475.065948
    depth_fy = 475.065857

    depth_camera_intr = camera.CameraIntr(
        **{"u0": depth_u0, "v0": depth_v0, "fx": depth_fx, "fy": depth_fy})

    # Extrinsic parameters
    # Rotations
    R = np.array([
        [0.999988496304, -0.00468848412856, 0.000982563360594],
        [0.00469115935266, 0.999985218048, -0.00273845880292],
        [-0.000969709653873, 0.00274303671904, 0.99999576807],
    ])
    # Translate
    t = np.array([25.7, 1.22, 3.902])

    camera_extr = camera.CameraExtr(R, t)

    cameras = {}
    cameras["rgb_camera_intr"] = rgb_camera_intr
    cameras["depth_camera_intr"] = depth_camera_intr
    cameras["camera_extr"] = camera_extr
    return cameras


CAMERAS = create_camera_matrix()
RGB_CAMERA_INTR = CAMERAS["rgb_camera_intr"]
DEPTH_CAMERA_INTR = CAMERAS["depth_camera_intr"]
CAMERA_EXTR = CAMERAS["camera_extr"]


class FHADBaseDataset(DatasetMixin):
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
        # expand dim so that depth.shape is (C,H,W)
        return np.expand_dims(depth, axis=0)

    def load_annotations(self, dataset_dir, debug=False):
        logger.info("> load annotations mode = {}".format(self.mode))
        logger.info("> dataset_dir = {}".format(dataset_dir))
        annotation_dir = os.path.join(dataset_dir, "Hand_pose_annotation_v1")
        video_dir = os.path.join(dataset_dir, "Video_files")
        logger.info("> load annotations")
        annotations = []
        for subject_id, action, seq_idx in itertools.product(*[SUBJECT_INDICES, ACTION_LIST, SEQ_INDICES]):
            if self.mode == "train":
                # Cross subject: training subjects are 1, 3, 4. The rest for test.
                if not seq_idx in [1, 2, 4]:
                    continue
            else:
                if not seq_idx in [3]:
                    continue
            skeleton_path = os.path.join(
                annotation_dir,
                "Subject_{}".format(subject_id),
                action,
                str(seq_idx),
                "skeleton.txt"
            )
            if not os.path.exists(skeleton_path):
                # skip missing data
                continue
            if os.path.getsize(skeleton_path) == 0:
                # some categories has empty file
                continue

            skeleton = np.loadtxt(skeleton_path)
            num_frame, _ = skeleton.shape
            step = 5 if self.mode == "train" else 50
            for idx in range(0, num_frame, step):
                frame_id = int(skeleton[idx][0])
                subject_dir = os.path.join(
                    video_dir, "Subject_{}".format(subject_id))
                rgb_path = os.path.join(
                    subject_dir,
                    action,
                    str(seq_idx),
                    "color",
                    "color_{:04d}.jpeg".format(frame_id)
                )
                depth_path = os.path.join(
                    subject_dir,
                    action,
                    str(seq_idx),
                    "depth",
                    "depth_{:04d}.png".format(frame_id)
                )

                if DATA_CONVENTION == "ZYX":
                    depth_joint = skeleton[frame_id][1:].reshape(-1, 3)
                    # xyz -> zyx
                    depth_joint = depth_joint[:, ::-1]
                    rgb_joint = CAMERA_EXTR.world_zyx2cam_zyx(depth_joint)
                else:
                    depth_joint = skeleton[frame_id][1:].reshape(-1, 3)
                    rgb_joint = CAMERA_EXTR.world_xyz2cam_xyz(depth_joint)

                rgb_joint = rgb_joint[INDEX_CONVERTER]
                rgb_joint = wrist2palm(rgb_joint)
                depth_joint = depth_joint[INDEX_CONVERTER]
                depth_joint = wrist2palm(depth_joint)
                example = {}
                example["hand_side"] = "right"
                example["rgb_path"] = rgb_path
                example["depth_path"] = depth_path
                example["depth_joint"] = depth_joint
                example["rgb_joint"] = rgb_joint
                annotations.append(example)

            if debug:
                # only read a few sample
                break
        logger.info("> Done loading annotations")
        return annotations


def get_base_dataset(dataset_dir, **kwargs):
    base = FHADBaseDataset(dataset_dir, **kwargs)
    return base


def get_fhad_dataset(dataset_dir, param, **kwargs):
    base = get_base_dataset(dataset_dir, **kwargs)
    dataset = HandDataset(base, param)
    return dataset


def get_dataset(dataset_dir, param, **kwargs):
    dataset = get_fhad_dataset(dataset_dir, param, **kwargs)
    return dataset


def visualize_dataset(dataset_dir):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # NOQA
    from chainercv.utils import read_image
    from visualizations import vis_point, vis_edge

    dataset = FHADBaseDataset(dataset_dir, debug=True, mode="train")
    example = dataset.get_example(0)
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
    dataset_dir = os.path.expanduser(
        "~/dataset/fhad")
    visualize_dataset(dataset_dir)
