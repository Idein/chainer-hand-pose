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

# +
import os
import pickle

import imageio
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from ipywidgets import interact
# -

# load annotations of this set
data_type = os.path.expanduser("~/dataset/RHD_published_v2/training")
with open(os.path.join(data_type, 'anno_training.pickle'), 'rb') as f:
    anno_all = pickle.load(f)

sample_id = 0
anno = anno_all[sample_id]

list(anno_all.keys())[:10]

file_format = "{:05d}.png".format(sample_id)
img_file = os.path.join(data_type, "color", file_format)
mask_file = os.path.join(data_type, "mask", file_format)
depth_file = os.path.join(data_type, "depth", file_format)


# auxiliary function
def depth_two_uint8_to_float(top_bits, bottom_bits):
    """ Converts a RGB-coded depth into float valued depth. """
    depth_map = (top_bits * 2**8 + bottom_bits).astype('float32')
    depth_map /= float(2**16 - 1)
    depth_map *= 5.0
    return depth_map


import cv2
depth = cv2.imread(depth_file)
# process rgb coded depth into float: top bits are stored in red, bottom in green channel
# depth in meters from the camera
depth = depth_two_uint8_to_float(depth[:, :, 0], depth[:, :, 1])

# get info from annotation dictionary
# u, v coordinates of 42 hand keypoints, pixel
kp_coord_uv = anno['uv_vis'][:, :2]
# visibility of the keypoints, boolean
kp_visible = (anno['uv_vis'][:, 2] == 1)
kp_coord_xyz = anno['xyz']  # x, y, z coordinates of the keypoints, in meters
camera_intrinsic_matrix = anno['K']  # matrix containing intrinsic parameters
# Project world coordinates into the camera frame
kp_coord_uv_proj = np.matmul(
    kp_coord_xyz, np.transpose(camera_intrinsic_matrix))
kp_coord_uv_proj = kp_coord_uv_proj[:, :2] / kp_coord_uv_proj[:, 2:]

from chainercv.utils import read_image
from chainercv.visualizations import vis_image

fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(111)
vis_image(read_image(img_file), ax=ax1)
ax1.plot(kp_coord_uv[kp_visible, 0], kp_coord_uv[kp_visible, 1], 'ro')
ax1.plot(kp_coord_uv_proj[kp_visible, 0],
         kp_coord_uv_proj[kp_visible, 1], 'gx')

fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(111)
ax1.imshow(depth)
ax1.plot(kp_coord_uv[kp_visible, 0], kp_coord_uv[kp_visible, 1], 'ro')
ax1.plot(kp_coord_uv_proj[kp_visible, 0],
         kp_coord_uv_proj[kp_visible, 1], 'gx')

vis_image(255*read_image(mask_file, dtype=np.uint8))

np.unique(cv2.imread(mask_file))

# +
# %matplotlib notebook

fig = plt.figure()
ax1 = fig.add_subplot(111, projection="3d")
ax1.scatter(kp_coord_xyz[kp_visible, 0],
            kp_coord_xyz[kp_visible, 1], kp_coord_xyz[kp_visible, 2])

# aligns the 3d coord with the camera view
ax1.view_init(azim=-90.0, elev=-90.0)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

# +
import itertools


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


"""
Keypoints available:
0: left wrist, 1-4: left thumb [tip to palm], 5-8: left index, ..., 17-20: left pinky,
21: right wrist, 22-25: right thumb, ..., 38-41: right pinky
"""

KEYPOINT_NAMES = [
    "wrist",
    "thumb_tip",
    "thumb_dip",
    "thumb_pip",
    "thumb_mcp",
    "index_tip",
    "index_dip",
    "index_pip",
    "index_mcp",
    "middle_tip",
    "middle_dip",
    "middle_pip",
    "middle_mcp",
    "ring_tip",
    "ring_dip",
    "ring_pip",
    "ring_mcp",
    "little_tip",
    "little_dip",
    "little_pip",
    "little_mcp",
]

ONESIDE_KEYPOINT_NAMES = []

for k in ["wrist", "thumb", "index", "middle", "ring", "little"]:
    if k == "wrist":
        joint_name = "_".join([k])
        ONESIDE_KEYPOINT_NAMES.append(joint_name)
    else:
        for p in ["tip", "dip", "pip", "mcp"]:
            joint_name = "_".join([k, p])
            ONESIDE_KEYPOINT_NAMES.append(joint_name)

assert KEYPOINT_NAMES == ONESIDE_KEYPOINT_NAMES

# (R,G,B)
BASE_COLOR = {
    "index": (0, 255, 0),
    "middle": (0, 0, 255),
    "ring": (255, 0, 255),
    "little": (255, 255, 0),
    "thumb": (255, 0, 0),
    "wrist": (50, 50, 50),
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
    i_s = KEYPOINT_NAMES.index(s)
    i_t = KEYPOINT_NAMES.index(t)
    COLOR_MAP[i_s, i_t] = COLOR_MAP[s, t]
    COLOR_MAP[KEYPOINT_NAMES.index(s)] = COLOR_MAP[s]
    COLOR_MAP[KEYPOINT_NAMES.index(t)] = COLOR_MAP[t]

# +
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib notebook
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

sample_id = 0
kp_xyz = anno_all[sample_id]["xyz"]
left_joints = kp_xyz[:21]
right_joints = kp_xyz[21:]
# aligns the 3d coord with the camera view
ax.view_init(azim=-90.0, elev=-90.0)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

for joints in [left_joints, right_joints]:
    tips = ["mcp" in kname or kname == "wrist" for kname in KEYPOINT_NAMES]
    xs = joints[:, 0][tips]
    ys = joints[:, 1][tips]
    zs = joints[:, 2][tips]
    color = np.array([COLOR_MAP[k] for k in KEYPOINT_NAMES])
    color = color[tips]/255.
    ax.scatter(xs, ys, zs, color=color)

    xs = joints[:, 0]
    ys = joints[:, 1]
    zs = joints[:, 2]
    for s, t in EDGES:
        sx = xs[s]
        sy = ys[s]
        sz = zs[s]
        tx = xs[t]
        ty = ys[t]
        tz = zs[t]
        color = COLOR_MAP[s, t]/255.
        ax.plot([sx, tx], [sy, ty], [sz, tz], color=color)

# +
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib notebook
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

sample_id = 0
kp_xyz = anno_all[sample_id]["xyz"]
left_joints = kp_xyz[:21]
right_joints = kp_xyz[21:]
# aligns the 3d coord with the camera view
ax.view_init(azim=-90.0, elev=-90.0)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

for joints in [left_joints, right_joints]:
    xs = joints[:, 0]
    ys = joints[:, 1]
    zs = joints[:, 2]
    color = [COLOR_MAP[k]/255. for k in KEYPOINT_NAMES]
    ax.scatter(xs, ys, zs, color=color)
    for s, t in EDGES:
        sx, sy, sz = xs[s], ys[s], zs[s]
        tx, ty, tz = xs[t], ys[t], zs[t]
        color = COLOR_MAP[s, t]/255.
        ax.plot([sx, tx], [sy, ty], [sz, tz], color=color)

# +
# get info from annotation dictionary
# u, v coordinates of 42 hand keypoints, pixel
kp_coord_uv = anno['uv_vis'][:, :2]
# visibility of the keypoints, boolean
kp_visible = (anno['uv_vis'][:, 2] == 1)
kp_coord_xyz = anno['xyz']  # x, y, z coordinates of the keypoints, in meters
camera_intrinsic_matrix = anno['K']  # matrix containing intrinsic parameters
# Project world coordinates into the camera frame

kp_coord_uv_proj = np.matmul(
    kp_coord_xyz, np.transpose(camera_intrinsic_matrix))
# divide x and y by z
zs = kp_coord_uv_proj[:, 2:]
xys = kp_coord_uv_proj[:, :2]
kp_coord_uv_proj = xys/zs
# -

from ipywidgets import interact

# +
# %matplotlib inline

# load annotations of this set
dataset_dir = os.path.expanduser("~/dataset/RHD_published_v2")
train_dir = os.path.join(dataset_dir, "training")
with open(os.path.join(train_dir, 'anno_training.pickle'), 'rb') as f:
    anno_all = pickle.load(f)


def visualize_dataset(sample_id):
    anno = anno_all[sample_id]
    file_format = "{:05d}.png".format(sample_id)
    img_file = os.path.join(train_dir, "color", file_format)
    mask_file = os.path.join(train_dir, "mask", file_format)
    depth_file = os.path.join(train_dir, "depth", file_format)
    kp_xyz = anno["xyz"]
    #print(anno["uv_vis"])
    left_kp_vis = (anno["uv_vis"][:21,2] == 1)
    right_kp_vis = (anno["uv_vis"][21:,2] == 1)
    left_joints = kp_xyz[:21]
    right_joints = kp_xyz[21:]

    kp_xyz = anno['xyz']  # x, y, z coordinates of the keypoints, in meters
    # matrix containing intrinsic parameters
    camera_intrinsic_matrix = anno['K']
    # Project world coordinates into the camera frame
    kp_coord_uv_proj = np.matmul(
        kp_xyz, np.transpose(camera_intrinsic_matrix))
    kp_coord_uv = kp_coord_uv_proj[:, :2] / kp_coord_uv_proj[:, 2:]
    left_uv = kp_coord_uv[:21]
    right_uv = kp_coord_uv[21:]

    # aligns the 3d coord with the camera view
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    depth = imageio.imread(depth_file)
    depth = depth_two_uint8_to_float(depth[:, :, 0], depth[:, :, 1])
    img = imageio.imread(img_file)
    ax1.imshow(img)
    ax2.imshow(depth)
    ax3 = fig.add_subplot(223, projection="3d")
    ax3.view_init(azim=-90.0, elev=-65.0)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')

    for joints,visible in zip([left_uv, right_uv],[left_kp_vis,right_kp_vis]):
        H, W, C = img.shape
        xs = joints[:, 0]
        ys = joints[:, 1]
        if not np.logical_and(np.all(0 < xs), np.all(xs < W)):
            continue
        if not np.logical_and(np.all(0 < ys), np.all(ys < H)):
            continue
        color = np.array([COLOR_MAP[k] for k in KEYPOINT_NAMES])/255.
        ax1.scatter(xs, ys, color=color)
        ax2.scatter(xs, ys, color=color)
        print(visible)
        ax1.scatter(xs[visible],ys[visible],marker='x')
    for joints in [left_joints, right_joints]:
        xs = joints[:, 0]
        ys = joints[:, 1]
        zs = joints[:, 2]
        color = np.array([COLOR_MAP[k] for k in KEYPOINT_NAMES])/255.
        ax3.scatter(xs, ys, zs, color=color)

        xs = joints[:, 0]
        ys = joints[:, 1]
        zs = joints[:, 2]
        for s, t in EDGES:
            sx = xs[s]
            sy = ys[s]
            sz = zs[s]
            tx = xs[t]
            ty = ys[t]
            tz = zs[t]
            color = COLOR_MAP[s, t]/255.
            ax3.plot([sx, tx], [sy, ty], [sz, tz], color=color)


interact(visualize_dataset, sample_id=range(1000, 3000, 10))
# -

np.logical_or(np.all(np.array([1, 2, 3]) > 2), np.all(np.array([1, 2, 3]) < 2))

# # Add keypoint noise

# +
# %matplotlib inline

# load annotations of this set
dataset_dir = os.path.expanduser("~/dataset/RHD_published_v2")
train_dir = os.path.join(dataset_dir, "training")
with open(os.path.join(train_dir, 'anno_training.pickle'), 'rb') as f:
    anno_all = pickle.load(f)


def visualize_dataset(sample_id):
    anno = anno_all[sample_id]
    file_format = "{:05d}.png".format(sample_id)
    img_file = os.path.join(train_dir, "color", file_format)
    mask_file = os.path.join(train_dir, "mask", file_format)
    depth_file = os.path.join(train_dir, "depth", file_format)
    kp_xyz = anno["xyz"]
    # print(anno["uv_vis"])
    left_kp_vis = (anno["uv_vis"][:21, 2] == 1)
    right_kp_vis = (anno["uv_vis"][21:, 2] == 1)
    left_joints = kp_xyz[:21]
    right_joints = kp_xyz[21:]

    # x, y, z coordinates of the keypoints, in meters
    kp_xyz = 1000 * anno['xyz']

    # add noise
    kp_xyz_noise = kp_xyz+np.random.normal(0, 1.22, (21*2, 3))

    # matrix containing intrinsic parameters
    camera_intrinsic_matrix = anno['K']
    # Project world coordinates into the camera frame
    kp_coord_uv_proj = np.matmul(
        kp_xyz, np.transpose(camera_intrinsic_matrix))
    kp_coord_uv = kp_coord_uv_proj[:, :2] / kp_coord_uv_proj[:, 2:]
    left_uv = kp_coord_uv[:21]
    right_uv = kp_coord_uv[21:]
    # noise
    kp_coord_uv_proj_noise = np.matmul(
        kp_xyz_noise, np.transpose(camera_intrinsic_matrix))
    kp_coord_uv_noise = kp_coord_uv_proj_noise[:,
                                               :2] / kp_coord_uv_proj_noise[:, 2:]
    left_uv_noise = kp_coord_uv_noise[:21]
    right_uv_noise = kp_coord_uv_noise[21:]
    # aligns the 3d coord with the camera view
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    depth = imageio.imread(depth_file)
    depth = depth_two_uint8_to_float(depth[:, :, 0], depth[:, :, 1])
    img = imageio.imread(img_file)
    ax1.imshow(img)
    ax2.imshow(depth)
    ax3 = fig.add_subplot(223, projection="3d")
    ax3.view_init(azim=-90.0, elev=-65.0)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    print(np.abs(left_uv-left_uv_noise).max(),np.abs(left_uv-left_uv_noise).min())
    for joints, joint_noise in zip([left_uv, right_uv], [left_uv_noise, right_uv_noise]):
        H, W, C = img.shape
        for i, j in enumerate([joints, joint_noise]):
            xs = j[:, 0]
            ys = j[:, 1]
            if not np.logical_and(np.all(0 < xs), np.all(xs < W)):
                continue
            if not np.logical_and(np.all(0 < ys), np.all(ys < H)):
                continue
            color = np.array([COLOR_MAP[k] for k in KEYPOINT_NAMES])/255.
            if i == 0:
                ax1.scatter(xs, ys, color=color)
                ax2.scatter(xs, ys, color=color)
            else:
                ax1.scatter(xs, ys, marker='x')

    for joints in [left_joints, right_joints]:
        xs = joints[:, 0]
        ys = joints[:, 1]
        zs = joints[:, 2]
        color = np.array([COLOR_MAP[k] for k in KEYPOINT_NAMES])/255.
        ax3.scatter(xs, ys, zs, color=color)

        xs = joints[:, 0]
        ys = joints[:, 1]
        zs = joints[:, 2]
        for s, t in EDGES:
            sx = xs[s]
            sy = ys[s]
            sz = zs[s]
            tx = xs[t]
            ty = ys[t]
            tz = zs[t]
            color = COLOR_MAP[s, t]/255.
            ax3.plot([sx, tx], [sy, ty], [sz, tz], color=color)


interact(visualize_dataset, sample_id=range(1000, 3000, 10))
