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

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
from scipy.io import loadmat

# +
dataset_dir=os.path.expanduser("~/dataset/stb/")
target="B1Counting"
image_dir = os.path.join(dataset_dir,"images")
label_dir = os.path.join(dataset_dir,"labels")

matBB = loadmat(os.path.join(label_dir, "_".join([target, "BB.mat"])))
# (xyz,joint_id,sample_id) -> (sample_id,joint_id,xyz)
annotationsBB = matBB["handPara"].transpose(2, 1, 0)

matSK = loadmat(os.path.join(label_dir, "_".join([target, "SK.mat"])))
# (xyz,joint_id,sample_id) -> (sample_id,joint_id,xyz)
annotationsSK = matSK["handPara"].transpose(2, 1, 0)
print(annotationsSK.shape)

# +
import itertools


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
# %matplotlib notebook

sample_id = 1120
joints = annotationsBB[sample_id]
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")


color = [COLOR_MAP[k]/255. for k in KEYPOINT_NAMES]
ax.scatter(*joints.transpose(), color=color)

for s, t in EDGES:
    color = COLOR_MAP[s, t]/255.
    ax.plot(*np.array([joints[s], joints[t]]).transpose(), color=color)

# +
# %matplotlib notebook

sample_id = 0
joints = annotationsSK[sample_id]
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

color = [COLOR_MAP[k]/255. for k in KEYPOINT_NAMES]
ax.scatter(*joints.transpose(), color=color)

for s, t in EDGES:
    color = COLOR_MAP[s, t]/255.
    ax.plot(*np.array([joints[s], joints[t]]).transpose(), color=color)

# +
import imageio
# %matplotlib inline


depth_file = os.path.join(image_dir, target, "SK_depth_{}.png".format(sample_id))
img_file = os.path.join(image_dir, target, "SK_color_{}.png".format(sample_id))
depth_image = imageio.imread(depth_file)
color_image = imageio.imread(img_file)
plt.imshow(color_image)

# +
import cv2
fx = 607.92271
fy = 607.88192
tx = 314.78337
ty = 236.42484
camera_intrinsic_mat = np.array(
    [
        [fx, 0, tx],
        [0, fy, ty],
        [0, 0,  1.],
    ]
)

joints = annotationsSK[sample_id]
rotation_vector = np.array([0.00531,   -0.01196,  0.00301])
translation_vector = np.array([-24.0381,   -0.4563,   -1.2326])
tv = translation_vector

fig, ax = plt.subplots()

uv, _ = cv2.projectPoints(
    joints,
    rotation_vector*(-1),
    translation_vector*(-1),
    camera_intrinsic_mat,
    np.zeros((5, 1)),
)
uv = np.squeeze(uv).transpose()
color = [COLOR_MAP[k]/255. for k in KEYPOINT_NAMES]
ax.scatter(*uv, color=color)
print(uv)
ax.imshow(color_image)

# +
import cv2
fx = 607.92271
fy = 607.88192
tx = 314.78337
ty = 236.42484
camera_intrinsic_mat = np.array(
    [
        [fx, 0, tx],
        [0, fy, ty],
        [0, 0,  1.],
    ]
)

joints = annotationsSK[sample_id]
rotation_vector = np.array([0.00531,   -0.01196,  0.00301])
translation_vector = np.array([-24.0381,   -0.4563,   -1.2326])
rot_mat, _ = cv2.Rodrigues(rotation_vector)

rxyz = (joints - translation_vector).dot(rot_mat)
uv_ = camera_intrinsic_mat.dot(rxyz.transpose())
uv = uv_/uv_[2:]
uv = uv[:2]
fig, ax = plt.subplots()
color = [COLOR_MAP[k]/255. for k in KEYPOINT_NAMES]
ax.scatter(*uv, color=color)
print(uv)
ax.imshow(color_image)

# +
# %matplotlib notebook

import cv2
fx = 475.62768
fy = 474.77709
tx = 336.41179
ty = 238.77962
camera_intrinsic_mat = np.array(
    [
        [fx, 0, tx],
        [0, fy, ty],
        [0, 0,  1.],
    ]
)

joints = annotationsSK[sample_id]
rotation_vector = np.array([0.00531,   -0.01196,  0.00301])
translation_vector = np.array([-24.0381,   -0.4563,   -1.2326])
rot_mat, _ = cv2.Rodrigues(rotation_vector)

rxyz = (joints).dot(rot_mat)
uv_ = camera_intrinsic_mat.dot(rxyz.transpose())
uv = uv_/uv_[2:]
uv = uv[:2]
fig, ax = plt.subplots()
color = [COLOR_MAP[k]/255. for k in KEYPOINT_NAMES]
ax.scatter(*uv, color=color)

depth_file = os.path.join(
    image_dir, target, "SK_depth_{}.png".format(sample_id))
img_file = os.path.join(image_dir, target, "SK_color_{}.png".format(sample_id))
depth_image = imageio.imread(depth_file)

depth_image = depth_image[:, :, 0]+256*depth_image[:, :, 1]
ax.imshow(depth_image, cmap="gray")
# -

rotation_vector = np.array([0.00531,   -0.01196,  0.00301])
translation_vector = np.array([-24.0381,   -0.4563,   -1.2326])
rot_mat, _ = cv2.Rodrigues(rotation_vector)

rot_mat

# +
import math

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

