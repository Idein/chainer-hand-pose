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

# +
import math
import os
import pickle

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
np.set_printoptions(precision=4, suppress=True)
# -

# load annotations of this set
data_type = os.path.expanduser("~/dataset/RHD_published_v2/training")
with open(os.path.join(data_type, 'anno_training.pickle'), 'rb') as f:
    anno_all = pickle.load(f)

sample_id = 128
anno = anno_all[sample_id]

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
# -

# # Rotate hand to create Canonical Representation

# We will treat left_joints only
kp_xyz = anno_all[sample_id]["xyz"]
hand_side = "left"

ROOT_JOINT = "wrist"
root_idx = ONESIDE_KEYPOINT_NAMES.index(ROOT_JOINT)
REFERENCE_MCP = "middle_mcp"
REFERENCE_PIP = "middle_pip"
ref_mcp_idx = ONESIDE_KEYPOINT_NAMES.index(REFERENCE_MCP)
ref_pip_idx = ONESIDE_KEYPOINT_NAMES.index(REFERENCE_PIP)
PINKY = "little_mcp"
little_mcp_idx = ONESIDE_KEYPOINT_NAMES.index(PINKY)
print(little_mcp_idx)
HAND_SIDE = "left"
if HAND_SIDE == "right":
    offset = len(KEYPOINT_NAMES)
    root_idx += offset
    ref_mcp_idx += offset
    ref_pip_idx += offset
    little_mcp_idx += offset


def get_oneside_hand(kp_xyz, hand_side):
    hand_side = "left"
    if hand_side == "left":
        joints = kp_xyz[:21]
    if hand_side == "right":
        joints = kp_xyz[21:]
    return joints


# ## normalize

def normalize_joints(joints):
    ref_length = np.linalg.norm(joints[ref_mcp_idx]-joints[ref_pip_idx])
    joints = (joints-joints[root_idx])/ref_length
    return joints


# # calc angle[rad] of rotation around z axis
#
# ```
# Y
# A
#  |     /
# y|----/
#  |θ /|
#  |__/ |
#  | /  | 
#  |/---x-----> X
# ```
#

def rot_z(theta):
    return np.array([
        [math.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), math.cos(theta), 0],
        [0, 0, 1],
    ])


# # calc angle[rad] of rotation around x axis
#
# ```
#  Z
#  A
#  |z_____/
#  |    / |
#  |   /  |
#  |  /   |
#  | /|θ | 
#  |/-|---y--> Y
# ```

def rot_x(theta):
    return np.array([
        [1, 0, 0],
        [0, math.cos(theta), -math.sin(theta)],
        [0, math.sin(theta), math.cos(theta)],
    ])


# # calc angle[rad] of rotation around y axis
#
# ```
#            Z
#            A
#      ------|z
#     | \    |
#     |  \   | 
#     |   \  | 
#     | θ| \| 
# <---|x--|-◎-
# ```

def rot_y(theta):
    return np.array([
        [math.cos(-theta), 0, -math.sin(-theta)],
        [0, 1, 0],
        [math.sin(-theta), 0, math.cos(-theta)],
    ])


def canonicalize_joints(joints):
    joints = normalize_joints(joints.copy())
    rot = np.eye(3)
    # rotate around z
    mcp_joint_x, mcp_joint_y, _ = joints[ref_mcp_idx]
    theta = math.atan2(mcp_joint_x, mcp_joint_y)
    joints = joints @ rot_z(theta).transpose()
    rot = rot_z(theta) @ rot
    # rotate around x
    _, mcp_joint_y, mcp_joint_z = joints[ref_mcp_idx]
    theta = math.atan2(mcp_joint_z, mcp_joint_y)
    # note that specify `-theta`. NOT `theta`
    joints = joints @ rot_x(-theta).transpose()
    rot = rot_x(-theta) @ rot
    # rotate around y
    mcp_joint_x, _, mcp_joint_z = joints[little_mcp_idx]
    theta = math.atan2(mcp_joint_z, -mcp_joint_x)
    # note that specify `-theta`. NOT `theta`
    joints = joints @ rot_y(-theta).transpose()
    rot = rot_y(-theta) @ rot
    mcp_joint_x, _, mcp_joint_z = joints[little_mcp_idx]
    return joints, rot


# # Let's Visualize

# +
from collections.abc import Iterable
# %matplotlib inline


def label_xyz(axis_object):
    if not isinstance(axis_object, Iterable):
        axes = [axis_object]
    else:
        axes = axis_object
        for ax in axis_object:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")


def visualize_canonical_representation(joints):
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")
    canonical_joints, rot = canonicalize_joints(joints.copy())
    normal_joints = normalize_joints(joints.copy())
    ax1.scatter(*canonical_joints.transpose())
    ax2.scatter(*joints.transpose())
    ax3.scatter(*normal_joints.transpose())
    # pullback canonical joints to normalized_joints
    pullback_joints = canonical_joints@np.linalg.inv(rot.transpose())
    ax3.scatter(*pullback_joints.transpose(), alpha=0.5)
    for s, t in EDGES:
        color = COLOR_MAP[s, t]/255.
        ax1.plot(*canonical_joints[[s, t]].transpose(), color=color)
        ax2.plot(*joints[[s, t]].transpose(), color=color)
        ax3.plot(*normal_joints[[s, t]].transpose(), color=color)
        ax3.plot(*pullback_joints[[s, t]].transpose(), alpha=0.5)
    ax1.view_init(0, -90)
    ax2.view_init(-90, -90)
    ax3.view_init(-90, -90)
    ax1.set_title("canonical")
    ax2.set_title("original")
    ax3.set_title("nomalized")
    label_xyz([ax1, ax2, ax3])


joints = get_oneside_hand(kp_xyz, hand_side=HAND_SIDE)
visualize_canonical_representation(joints)
# -

import cv2

mat, _ = cv2.Rodrigues(np.array([1., 2., 3.]))
mat.shape


def extract_euler_angles(mat):
    """
    This algorith is aken from
    Extracting Euler Angles from a Rotation Matrix
    Mike Day, Insomniac Games
    mday@insomniacgames.com
    https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2012/07/euler-angles1.pdf
    
    The authors follow the notational conventions of Shoemake’s “Euler Angle Conversion”, Graphics Gems IV, pp.
    222-9, with the exception that their vectors are row vectors instead of column vectors. Thus, all their
    matrices are transposed relative to Shoemake’s, and a sequence of rotations will be written from left to
    right. 
    """
    [[m00, m01, m02],
     [m10, m11, m12],
     [m20, m21, m22]] = mat
    theta_x = math.atan2(m12, m22)
    c2 = math.sqrt(m00**2+m01**2)
    theta_y = math.atan2(-m02, c2)
    s1 = math.sin(theta_x)
    c1 = math.cos(theta_x)
    theta_z = math.atan2(s1*m20-c1*m10, c1*m11-s1*m21)
    """
    multiply minus one for each theta_*. this is equivalent to RE-consider vector is column instead of rows
    i.e. back to common world that says vector is column.
    """
    return -theta_x, -theta_y, -theta_z


# +
# Test code

mat = rot_x(math.pi/3)
expected = mat.copy()
radians = extract_euler_angles(mat)
mat, _ = cv2.Rodrigues(np.asarray(radians))
assert np.allclose(mat, expected)

mat = rot_y(math.pi/3)
expected = mat.copy()
radians = extract_euler_angles(mat)
mat, _ = cv2.Rodrigues(np.asarray(radians))
assert np.allclose(mat, expected)

mat = rot_z(math.pi/3)
expected = mat.copy()
radians = extract_euler_angles(mat)
mat, _ = cv2.Rodrigues(np.asarray(radians))
assert np.allclose(mat, expected)

mat = rot_z(-math.pi/6) @ rot_z(2*math.pi/3) @ rot_z(-math.pi/3)
expected = mat.copy()
radians = extract_euler_angles(mat)
mat, _ = cv2.Rodrigues(np.asarray(radians))
assert np.allclose(mat, expected)

# +
# differential rodorigues

# +
import numpy as xp

def variable_rodrigues(rotation_vector):
    B = rotation_vector.shape[0]
    xp = get_array_module(rotation_vector)
    theta = F.sqrt(F.batch_l2_norm_squared(rotation_vector))
    rv = rotation_vector/F.expand_dims(theta, axis=1)
    v1 = F.expand_dims(rv, axis=2)
    v2 = F.expand_dims(rv, axis=1)
    rr = F.matmul(v1, v2)
    R = F.cos(theta).reshape(B, 1, 1) * xp.array(B*[xp.eye(3)])
    R += (1-F.cos(theta)).reshape(B, 1, 1) * rr
    # define roation vector
    nu=xp.zeros(B,xp.float32)
    xs = rv[:,0]
    ys = rv[:,1]
    zs = rv[:,2]
    R += F.sin(theta).reshape(B, 1, 1) * F.stack([
         nu,-zs,ys,\
         zs,nu,-xs,\
         -ys,xs,nu
    ],axis=1).reshape(-1,3,3)
    return R


# -

def variable_rodrigues(rotation_vector):
    """
    Calculate rotation matrix for variable of Chainer.
    This implementation outputs the same as the following code:
    def myRodrigues(rotation_vector):
        theta = np.linalg.norm(rotation_vector)
        rv = rotation_vector/theta
        rr = np.array([[rv[i]*rv[j] for j in range(3)] for i in range(3)])
        R = math.cos(theta) * np.eye(3)
        R += (1-math.cos(theta)) * rr
        R += math.sin(theta) * np.array([
            [0, -rv[2], rv[1]],
            [rv[2], 0, -rv[0]],
            [-rv[1], rv[0], 0],
        ])
        return R
    """
    B = rotation_vector.shape[0]
    xp = get_array_module(rotation_vector)
    theta = F.sqrt(F.batch_l2_norm_squared(rotation_vector))
    rv = rotation_vector/F.expand_dims(theta, axis=1)
    theta = theta.reshape(B,1,1)
    rr = F.matmul(F.expand_dims(rv, axis=2), F.expand_dims(rv, axis=1))
    R = F.cos(theta) * xp.array(B*[xp.eye(3)])
    R += (1-F.cos(theta)).reshape(B, 1, 1) * rr
    nu=xp.zeros(B,xp.float32)
    xs,ys,zs = rv[:,0],rv[:,1],rv[:,2]
    R += F.sin(theta) * F.stack([
         nu,-zs,ys,\
         zs,nu,-xs,\
         -ys,xs,nu
    ],axis=1).reshape(-1,3,3)
    return R


import chainer
import chainer.functions as F
from chainer.backends.cuda import get_array_module
vec=chainer.Variable(np.array([[1,2,3],[4,5,6]],np.float32))
vrod=variable_rodrigues(vec)


def myRodrigues(rotation_vector):
    theta = np.linalg.norm(rotation_vector)
    rv = rotation_vector/theta
    rr = np.array([[rv[i]*rv[j] for j in range(3)] for i in range(3)])
    R = math.cos(theta) * np.eye(3)
    R += (1-math.cos(theta)) * rr
    R += math.sin(theta) * np.array([
        [0, -rv[2], rv[1]],
        [rv[2], 0, -rv[0]],
        [-rv[1], rv[0], 0],
    ])
    return R


assert np.allclose(vrod[0].array, myRodrigues(np.array([1,2,3])))
assert np.allclose(vrod[1].array, myRodrigues(np.array([4,5,6])))
