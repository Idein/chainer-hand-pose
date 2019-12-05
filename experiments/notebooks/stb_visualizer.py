# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # stb_visualizer
#
# https://github.com/zhjwustc/icip17_stereo_hand_pose_dataset

# +
import os
import itertools

import imageio
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
from scipy.io import loadmat

from ipywidgets import interact


# +
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


KEYPOINT_NAMES = [
    "wrist",
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

# (R,G,B)
BASE_COLOR = {
    "wrist": (50, 50, 50),
    "index": (0, 255, 0),
    "middle": (0, 0, 255),
    "ring": (255, 0, 255),
    "little": (255, 255, 0),
    "thumb": (255, 0, 0),
}

# convert tuple to numpy array
BASE_COLOR = {k: np.array(v) for k, v in BASE_COLOR.items()}

COLOR_MAP = {"wrist": BASE_COLOR["wrist"]}
EDGES_BY_NAME = []

for f in ["index", "middle", "ring", "little", "thumb"]:
    for p, q in pairwise(["wrist", "mcp", "pip", "dip", "tip"]):
        color = BASE_COLOR[f]
        if p != "wrist":
            p = "_".join([f, p])
        q = "_".join([f, q])
        COLOR_MAP[p, q] = color
        COLOR_MAP[q] = color
        EDGES_BY_NAME.append([p, q])

EDGES = [[KEYPOINT_NAMES.index(s), KEYPOINT_NAMES.index(t)]
         for s, t in EDGES_BY_NAME]

for s, t in EDGES_BY_NAME:
    s_i = KEYPOINT_NAMES.index(s)
    t_i = KEYPOINT_NAMES.index(t)
    COLOR_MAP[s_i, t_i] = COLOR_MAP[s, t]
    COLOR_MAP[KEYPOINT_NAMES.index(s)] = COLOR_MAP[s]
    COLOR_MAP[KEYPOINT_NAMES.index(t)] = COLOR_MAP[t]

# +
import cv2
fx = 607.92271
fy = 607.88192
tx = 314.78337
ty = 236.42484
camera_intrinsic_mat = np.array([
    [fx, 0, tx],
    [0, fy, ty],
    [0, 0,  1.],
])

rotation_vector = np.array([0.00531,   -0.01196,  0.00301])
translation_vector = np.array([-24.0381,   -0.4563,   -1.2326])
rot_mat, _ = cv2.Rodrigues(rotation_vector)

# +
dataset_dir = os.path.expanduser("~/dataset/stb/")
image_dir = os.path.join(dataset_dir, "images")
label_dir = os.path.join(dataset_dir, "labels")

# %matplotlib inline


def visualize_stb(seq, frame_idx):
    matBB = loadmat(
        os.path.join(label_dir, "_".join([seq, "BB.mat"])))
    # (xyz,joint_id,frame_idx) -> (frame_idx,joint_id,xyz)
    annotationsBB = matBB["handPara"].transpose(2, 1, 0)

    matSK = loadmat(
        os.path.join(label_dir, "_".join([seq, "SK.mat"])))
    # (xyz,joint_id,frame_idx) -> (frame_idx,joint_id,xyz)
    annotationsSK = matSK["handPara"].transpose(2, 1, 0)
    world_joints = annotationsSK[frame_idx]

    rgb_joints = (world_joints - translation_vector).dot(rot_mat)
    rgb_uv_ = camera_intrinsic_mat.dot(rgb_joints.transpose())
    rgb_uv = rgb_uv_/rgb_uv_[2:]
    rgb_uv = rgb_uv[:2]

    depth_joints = (world_joints).dot(rot_mat)
    depth_uv_ = camera_intrinsic_mat.dot(depth_joints.transpose())
    depth_uv = depth_uv_/depth_uv_[2:]
    depth_uv = depth_uv[:2]

    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222, projection="3d")

    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224, projection="3d")

    color = [COLOR_MAP[k]/255. for k in KEYPOINT_NAMES]
    ax1.scatter(*rgb_uv, color=color)
    ax3.scatter(*depth_uv, color=color)
    ax2.scatter(*rgb_joints.transpose(), color=color)
    ax4.scatter(*depth_joints.transpose(), color=color)
    edge_color = [COLOR_MAP[s, t]/255. for s, t in EDGES]

    for ((s, t), c) in zip(EDGES, edge_color):
        ax2.plot(*rgb_joints[[s, t]].transpose(), color=c)
        ax4.plot(*depth_joints[[s, t]].transpose(), color=c)

    depth_file = os.path.join(
        image_dir, seq, "SK_depth_{}.png".format(frame_idx))
    img_file = os.path.join(
        image_dir, seq, "SK_color_{}.png".format(frame_idx))
    color_image = imageio.imread(img_file)
    depth_image = imageio.imread(depth_file)

    ax1.imshow(color_image)
    ax3.imshow(depth_image)
    for ax in [ax2, ax4]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(-65, -90)


backgrounds = [1, 2, 3, 4, 5, 6]
frames = list(range(0, 1500, 20))

sequences = ["B{}Counting".format(i) for i in backgrounds]
sequences += ["B{}Random".format(i) for i in backgrounds]

interact(visualize_stb, seq=sequences, frame_idx=frames)
# -




