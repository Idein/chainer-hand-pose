# -*- coding: utf-8 -*-
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

# # first_person_action_cvpr2018_loader
#
# - https://github.com/guiggh/hand_pose_action

# # import modules

# +
import os

import numpy as np
from matplotlib import pyplot as plt

from ipywidgets import interact
# -

# # setting directories

dataset_dir = os.path.expanduser(
    "~/dataset/fhad")

video_dir = os.path.join(dataset_dir, "Video_files")

# +

subject_id = 1
subject_dir = os.path.join(video_dir, "Subject_{}".format(subject_id))
actions = [d.name for d in os.scandir(subject_dir)]
print(sorted(actions))
# -

action = actions[4]
print(action)
# 1,2,3, or 4
seq_idx = 1

# # visualize images

video_dir = os.path.join(subject_dir, action, str(seq_idx))

# +
import imageio

frame_idx = 0
img_path = os.path.join(
    video_dir,
    "color", "color_{:04d}.jpeg".format(frame_idx)
)
plt.imshow(imageio.imread(img_path))

# +
import imageio

idx = 0
depth_path = os.path.join(video_dir, "depth", "depth_{:04d}.png".format(idx))
plt.imshow(imageio.imread(depth_path), cmap="gray")
# -

# # Get annotation

annotation_dir = os.path.join(dataset_dir, "Hand_pose_annotation_v1")
skeleton_path = os.path.join(annotation_dir, "Subject_{}".format(
    subject_id), action, str(seq_idx), "skeleton.txt")

import pandas as pd

annotations = np.loadtxt(skeleton_path)
frame_indice,annotations = annotations[:,0],annotations[:, 1:]
joints=annotations[frame_idx].reshape(-1,3).transpose()

# +
# ’T’, ’I’, ’M’, ’R’, ’P’ denote ’Thumb’, ’Index’, ’Middle’, ’Ring’, ’Pinky’ fingers.
KEYPOINT_NAMES = [
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
for kname in KEYPOINT_NAMES:
    for c in ["X", "Y", "Z"]:
        cname = "_".join([c, kname])
        COLUMN_NAMES.append(cname)

# +
from itertools import tee


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


EDGE_NAMES = []
for p in ["T", "I", "M", "R", "P"]:
    for s, t in pairwise(["Wrist", "MCP", "PIP", "DIP", "TIP"]):
        if s == "Wrist":
            EDGE_NAMES.append([s, p+t])
        else:
            EDGE_NAMES.append([p+s, p+t])

EDGES = [[KEYPOINT_NAMES.index(s), KEYPOINT_NAMES.index(t)]
         for s, t in EDGE_NAMES]
# -

from mpl_toolkits.mplot3d import Axes3D

# # Overlay image and projected points

# %matplotlib notebook
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(*joints)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.view_init(-64,-90)
for s, t in EDGES:
    plt.plot(*joints[:, [s, t]])

# +
# Image center
u0 = 935.732544
v0 = 540.681030
# Focal Length
fx = 1395.749023
fy = 1395.749268

cam_extr = np.array([
    [0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
    [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
    [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
    [0, 0, 0, 1],
])

cam_intr = np.array([
    [fx, 0, u0],
    [0, fy, v0],
    [0, 0, 1],
])

# +
fig, ax = plt.subplots()
ax.imshow(imageio.imread(img_path))
skel = joints.transpose()
skel_hom = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
skel_camcoords = cam_extr.dot(skel_hom.transpose())
skel_camcoords = skel_camcoords[:3, :].astype(np.float32)
skel_hom2d = np.array(cam_intr).dot(skel_camcoords)
skel_proj = (skel_hom2d / skel_hom2d[2:])[:2]

ax.scatter(*skel_proj)

# +
# Image center:
u0 = 315.944855
v0 = 245.287079
# Focal Length:
fx = 475.065948
fy = 475.065857

dtype=np.float32
depth_cam_intr = np.array([
    [fx, 0, u0],
    [0, fy, v0],
    [0, 0, 1],
])

fig, ax = plt.subplots()
u, v, z = np.array(depth_cam_intr).dot(joints)
u = u/z
v = v/z
ax.imshow(imageio.imread(depth_path))
ax.scatter(u, v, color="r")

# +
pull_back_xyz = np.array([
    [1/fx, 0, -u0/fx],
    [0, 1/fy, -v0/fy],
    [0, 0, 1],
], dtype=dtype).transpose()

def label_3d(ax):
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

# taken from nyu_dataset_visualizer
def convert_depth_to_uvd(depth):
    H, W = depth.shape
    uv = np.meshgrid(range(W), range(H))
    uvd = np.concatenate([uv, np.expand_dims(depth, axis=0)], axis=0)
    return uvd

def uv2xyz(uv, z):
    nk, *_ = uv.shape
    hom_uv = np.concatenate([uv, np.ones((nk, 1), dtype=dtype)], axis=1)
    xy_ = hom_uv @ pull_back_xyz
    xyz = z*xy_
    return xyz


# +
from PIL import Image
# imageio.imread でも uint16 でのデータとしてデコードしてくれる
depth=np.asarray(Image.open(depth_path))
uvd=convert_depth_to_uvd(depth).transpose(1,2,0).reshape(-1,3)
uv,d=uvd[:,:2],uvd[:,2:]

H,W=depth.shape
print(H,W)
xyz=uv2xyz(uv,d).reshape(H,W,3).transpose(2,0,1)
sample_xyz=xyz[:,::15,::15]
fig=plt.figure()
ax=fig.add_subplot(111,projection="3d")
label_3d(ax)
ax.scatter(*joints,c="r")
ax.scatter(*sample_xyz,alpha=0.1)
ax.view_init(-115,-90)
