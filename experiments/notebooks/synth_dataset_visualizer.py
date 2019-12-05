# ---
# jupyter:
#   jupytext:
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

# # Load SynthHands Dataset

import os

dataset_dir = os.path.expanduser("~/dataset/SynthHands_Release")

# !cat $dataset_dir/README.txt

# !cat $dataset_dir/CameraCalibration_640x480.txt

# +
from glob import glob

import imageio
import numpy as np
import chainercv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from ipywidgets import interact

# +
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

# +
# %matplotlib notebook

SEX = ["male", "female"]
OBJ_SUFFIX = ["object", "noobject"]
SEQUENCES = [
    "seq01", "seq02", "seq03", "seq04", "seq05", "seq06", "seq07", ]
CAMERAS = ["cam01", "cam02", "cam03", "cam04", "cam05"]
PARTITIONS = ["01", "02", "03"]

rgb_camera_intr = np.array([
    [617.173, 0, 315.453],
    [0, 617.173, 242.259],
    [0, 0, 1],
])


USE_COLOR=True
USE_COLOR_ON_DEPTH=False


translation = np.array([24.7,  -0.0471401,  3.72045])


def visualize_dataset(sex, suffix, seq, cam, partition, frame):
    image_dir = os.path.join(
        dataset_dir,
        "_".join([sex, suffix]),
        seq,
        cam,
        partition,
    )
    color_path = os.path.join(
        image_dir, "{:08d}_color.png".format(frame)
    )
    color_on_depth_path = os.path.join(
        image_dir, "{:08d}_color_on_depth.png".format(frame)
    )
    depth_path = os.path.join(
        image_dir, "{:08d}_depth.png".format(frame)
    )
    joint_path = os.path.join(
        image_dir, "{:08d}_joint_pos.txt".format(frame)
    )

    if USE_COLOR:
        color = chainercv.utils.read_image(color_path)    
    if USE_COLOR_ON_DEPTH:
        color = chainercv.utils.read_image(color_on_depth_path)    
    world_joint = np.loadtxt(joint_path, delimiter=',').reshape(-1, 3)


    img_files = glob(os.path.join(image_dir, "*.png"))
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222, projection="3d")
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224, projection="3d")
    assert os.path.exists(color_path), color_path
    chainercv.visualizations.vis_image(color, ax=ax1)
    color = np.array([COLOR_MAP[k] for k in KEYPOINT_NAMES])/255.
    if USE_COLOR:
        rgb_joint3D = world_joint + translation
    if USE_COLOR_ON_DEPTH:
        rgb_joint3D = world_joint
    joint2D_hom = rgb_joint3D.dot(rgb_camera_intr.transpose())
    joint2D_hom = joint2D_hom/joint2D_hom[:, 2:]
    joint2D = joint2D_hom[:, :2]
    ax1.scatter(*joint2D.transpose(), color=color)
    ax2.scatter(*rgb_joint3D.transpose(), color=color)
    depth_joint3D = world_joint 
    joint2D_hom = rgb_joint3D.dot(rgb_camera_intr.transpose())
    joint2D_hom = joint2D_hom/joint2D_hom[:, 2:]
    joint2D = joint2D_hom[:, :2]
    for s, t in EDGES:
        color = COLOR_MAP[s, t]/255.
        ax1.plot(*joint2D[[s, t]].transpose(), c=color)
        ax2.plot(*rgb_joint3D[[s, t]].transpose(), c=color)
    depth = imageio.imread(depth_path)
    ax3.imshow(depth)
    ax3.scatter(*joint2D.transpose(),color=color)
    ax4.scatter(*depth_joint3D.transpose(),color=color)
    for s, t in EDGES:
        color = COLOR_MAP[s, t]/255.
        ax3.plot(*joint2D[[s, t]].transpose(), c=color)
        ax4.plot(*depth_joint3D[[s, t]].transpose(), c=color)
    for ax in [ax2,ax4]:
        ax.view_init(-65, -90)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")


interact(visualize_dataset, sex=SEX, suffix=OBJ_SUFFIX,
         seq=SEQUENCES, cam=CAMERAS, partition=PARTITIONS, frame=range(0, 500))
