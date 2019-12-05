# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Load GANerated Hands Dataset

# +
import os

dataset_dir = os.path.expanduser(
    "~/dataset/GANeratedHands_Release/data")

# +
import numpy as np
from ipywidgets import interact
import chainercv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


# +
KEYPOINT_NAMES = [
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

# (R,G,B)
BASE_COLOR = {
    "I": (0, 255, 0),
    "M": (0, 0, 255),
    "R": (255, 0, 255),
    "L": (255, 255, 0),
    "T": (255, 0, 0),
    "W": (50, 50, 50),
}

# convert tuple to numpy array
BASE_COLOR = {k: np.array(v) for k, v in BASE_COLOR.items()}

COLOR_MAP = {"W": BASE_COLOR["W"]}
EDGE_NAMES = []

for f in ["I", "M", "R", "L", "T"]:
    for p, q in pairwise(["W", "0", "1", "2", "3"]):
        color = BASE_COLOR[f]
        if p != "W":
            p = "".join([f, p])
        q = "".join([f, q])
        COLOR_MAP[p, q] = color
        COLOR_MAP[q] = color
        EDGE_NAMES.append([p, q])

EDGES = [[KEYPOINT_NAMES.index(s), KEYPOINT_NAMES.index(t)]
         for s, t in EDGE_NAMES]

for s, t in EDGE_NAMES:
    i_s = KEYPOINT_NAMES.index(s)
    i_t = KEYPOINT_NAMES.index(t)
    COLOR_MAP[i_s, i_t] = COLOR_MAP[s, t]
    COLOR_MAP[KEYPOINT_NAMES.index(s)] = COLOR_MAP[s]
    COLOR_MAP[KEYPOINT_NAMES.index(t)] = COLOR_MAP[t]

# convert value as np.array
COLOR_MAP = {k: np.array(v) for k, v in COLOR_MAP.items()}
# define root joint and reference edge. They are used to normalize 3D joint
ROOT_IDX = KEYPOINT_NAMES.index("W")
MMCP = KEYPOINT_NAMES.index("M0")
MPIP = KEYPOINT_NAMES.index("M1")
REF_EDGE = (MMCP, MPIP)
# -

NO_OBJ_DIR = os.listdir(os.path.join(dataset_dir, "noObject"))
WITH_OBJ_DIR = os.listdir(os.path.join(dataset_dir, "withObject"))

# See README.txt
camera_intr = np.array([
    [617.173, 0, 315.453],
    [0, 617.173, 242.259],
    [0,       0,       1],
]).transpose()

# +
OBJECTS = ["noObject", "withObject"]
MAX_FRAME_ID = 1024


def visualize_noobject(partition, frame_idx):
    img_dir = os.path.join(dataset_dir, "noObject", partition)
    fmt = "{:04d}".format(frame_idx)
    color_composed_path = os.path.join(img_dir, fmt+"_" + "color_composed.png")

    crop_params_path = os.path.join(img_dir, fmt+"_" + "crop_params.txt")
    joint2D_path = os.path.join(img_dir, fmt+"_" + "joint2D.txt")
    joint_pos_path = os.path.join(img_dir, fmt+"_" + "joint_pos.txt")
    joint_pos_global_path = os.path.join(
        img_dir, fmt+"_" + "joint_pos_global.txt")
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222, projection="3d")
    #ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224, projection="3d")
    assert os.path.exists(color_composed_path), color_composed_path
    rgb = chainercv.utils.read_image(color_composed_path)
    joint3D_normed = np.loadtxt(joint_pos_path, delimiter=',').reshape(-1, 3)
    # disable runtime warning zerodivision
    joint3D_normed +=1
    joint3D = np.loadtxt(joint_pos_global_path, delimiter=',').reshape(-1, 3)
    crop_u, crop_v, scale = crop_param = np.loadtxt(
        crop_params_path, delimiter=','
    )

    chainercv.visualizations.vis_image(rgb, ax=ax1)
    ax2.scatter(*joint3D.transpose())
    ax4.scatter(*joint3D_normed.transpose())

    joint2D = joint3D.dot(camera_intr)/joint3D[:, 2:]
    joint2D = joint2D[:, :2]
    joint2D = scale*(joint2D-np.array([[crop_u, crop_v]]))

    joint2D_normed = joint3D_normed.dot(camera_intr)/joint3D_normed[:, 2:]
    joint2D_normed = joint2D_normed[:, :2]
    
    joint2D_as_data=np.loadtxt(joint2D_path,delimiter=',').reshape(-1,2)
    ax1.scatter(*joint2D_as_data.transpose())
    ax1.scatter(*joint2D.transpose())
    #ax3.scatter(*joint2D_normed.transpose())

    for s, t in EDGES:
        ax1.plot(*joint2D[[s, t]].transpose())
        ax2.plot(*joint3D[[s, t]].transpose())
        #ax3.plot(*joint2D_normed[[s, t]].transpose())
        ax4.plot(*joint3D_normed[[s, t]].transpose())
    for ax in [ax2, ax4]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(-65, -90)


interact(visualize_noobject, partition=NO_OBJ_DIR, frame_idx=range(1, 1024+1))
# -


