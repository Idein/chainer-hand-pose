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

# # NYU dataset visualizer

# # import modules

# +
import os
from glob import glob

import imageio
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ipywidgets import interact
# -

import itertools
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


dataset_dir = os.path.expanduser("~/dataset/nyu_hand_dataset_v2")

# # check train data

train_kinect_id = [1, 2, 3]
for k in train_kinect_id:
    fmt = os.path.join(dataset_dir, "dataset", "train", "{0}_{1}_*.png")
    depth_ptn = fmt.format("depth", k)
    print(len(sorted(glob(depth_ptn))))

# # check test data

test_kinect_id = [1, 2, 3]
for k in test_kinect_id:
    fmt = os.path.join(dataset_dir, "dataset", "test", "{0}_{1}_*.png")
    depth_ptn = fmt.format("depth", k)
    print(sorted(glob(depth_ptn))[0])


# # Viualize image, depth
#  - Note: In each depth png file the top 8 bits of depth are packed into the green channel and the lower 8 bits into blue.

def decode_depthimg(depthimg):
    """
    depthimg.shape should be (H,W,C) where H is height, W is width, C is RGB channel i.e. C=3
    In each depth png file the top 8 bits of depth are packed into the green channel and the lower 8 bits into blue.
    """
    r = depthimg[:, :, 0]
    g = depthimg[:, :, 1].astype(np.uint16)
    b = depthimg[:, :, 2].astype(np.uint16)
    top_bit = 2**8 * g
    lower_bit = b
    depth = top_bit+lower_bit
    dpt = np.bitwise_or(np.left_shift(g, 8), b)
    assert (depth == dpt).all()
    print(depth.max(), g.max(), b.max())
    return depth

# +
# %matplotlib notebook


def visualize_image(k, f, is_train=False):
    fig = plt.figure(figsize=(5, 15))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    folder = "train" if is_train else "test"
    fmt = os.path.join(dataset_dir, "dataset", folder, "{0}_{1}_{2:07d}.png")
    rgb_file = fmt.format("rgb", k, f)
    depth_file = fmt.format("depth", k, f)
    synthdepth_file = fmt.format("synthdepth", k, f)
    print(synthdepth_file)
    ax1.imshow(imageio.imread(rgb_file))
    ax2.imshow(decode_depthimg(imageio.imread(depth_file)))
    ax3.imshow(decode_depthimg(imageio.imread(synthdepth_file)))


interact(visualize_image, k=[1, 2, 3], f=range(2, 100+1), is_train=False)
# -

# # load annotation

# +
# see convert_depth_to_uvd.m


def convert_depth_to_uvd(depth):
    H, W = depth.shape
    uv = np.meshgrid(range(W), range(H))
    uvd = np.concatenate([uv, np.expand_dims(depth, axis=0)], axis=0)
    return uvd


# see convert_uvd_to_xyz.m

xRes = 640
yRes = 480
xzFactor = 1.08836710
yzFactor = 0.817612648


def convert_uvd_to_xyz(uvd):
    # uvd.shape is (N,3) where N is num of keypoints
    normalizedX = uvd[:, 0] / xRes - 0.5
    normalizedY = 0.5 - uvd[:, 1] / yRes
    xyz = np.zeros(uvd.shape)
    xyz[:, 2] = uvd[:, 2]
    xyz[:, 0] = normalizedX * xyz[:, 2] * xzFactor
    xyz[:, 1] = normalizedY * xyz[:, 2] * yzFactor
    return xyz

# see convert_xyz_to_uvd.m


halfResX = 640/2
halfResY = 480/2
coeffX = 588.036865
coeffY = 587.075073

# check relation ship
assert np.allclose(np.array([xzFactor, yzFactor]),
                   np.array([xRes/coeffX, yRes/coeffY]))


def convert_xyz_to_uvd(xyz):
    uvd = np.zeros(xyz.shape)
    uvd[:, 0] = coeffX * xyz[:, 0] / xyz[:, 2] + halfResX
    uvd[:, 1] = halfResY - coeffY * xyz[:, 1] / xyz[:, 2]
    uvd[:, 2] = xyz[:, 2]
    return uvd


# -

from scipy.io import loadmat
matfile = os.path.join(dataset_dir, "dataset", "train", "joint_data.mat")
mat = loadmat(matfile, squeeze_me=True)

mat.keys()

KEYPOINT_NAMES = list(mat["joint_names"])
print(KEYPOINT_NAMES)

mat["joint_uvd"].shape

mat["joint_xyz"].shape

# +
# %matplotlib notebook


def label_xyz2d(ax):
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def label_xyz3d(ax):
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

# TODO:Visualize edge


def visualize_keypoint(k, f):
    fig = plt.figure(figsize=(5, 10))
    fmt = os.path.join(dataset_dir, "dataset", "train", "{0}_{1}_{2:07d}.png")
    rgb_file = fmt.format("rgb", k, f)
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313, projection="3d")
    img = imageio.imread(rgb_file)
    label_xyz2d(ax2)
    label_xyz3d(ax3)
    us, vs, ds = mat["joint_uvd"][k-1][f].transpose()
    ax1.imshow(img)
    ax1.scatter(us, vs)
    ax2.scatter(us, vs)
    joints = mat["joint_xyz"][k-1][f].transpose()
    xs, ys, zs = joints
    ax3.view_init(-90, -90)
    ax3.scatter(xs, ys, zs)
    xyz = convert_uvd_to_xyz(mat["joint_uvd"][k-1][f]).transpose()
    ax3.scatter(*xyz, color="r")
    uvd = convert_xyz_to_uvd(joints.transpose()).transpose()
    us, vs, _ = uvd
    ax2.scatter(us, vs, alpha=0.5)
    ax2.invert_yaxis()
    ax3.view_init(-60, 90)
    ax3.invert_xaxis()
    # ax2.set_aspect("equal")
    # ax3.set_aspect("equal")
    fig.tight_layout()


interact(visualize_keypoint, k=[1, 2, 3], f=range(2, 100+1))

# +
# %matplotlib notebook

# TODO:Visualize edge


def visualize_uvd_xyz(k, f):
    fig = plt.figure(figsize=(8, 10))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312, projection="3d")
    ax3 = fig.add_subplot(313, projection="3d")
    label_xyz3d(ax2)
    label_xyz3d(ax3)

    fmt = os.path.join(dataset_dir, "dataset", "train", "{0}_{1}_{2:07d}.png")
    rgb_file = fmt.format("rgb", k, f)
    depth_file = fmt.format("depth", k, f)

    img = imageio.imread(rgb_file)
    depth = decode_depthimg(imageio.imread(depth_file))
    uvd = convert_depth_to_uvd(depth)
    _, H, W = uvd.shape
    # fit interface
    # d,H,W -> H,W,d -> [N,3], where N=H*W, and axis=1 contains [u,v,d]
    xyz = convert_uvd_to_xyz(uvd.transpose(1, 2, 0).reshape(-1, 3))
    # restore shape
    # N,3 -> H,W,3 -> 3,H,W
    xyz = xyz.reshape(H, W, 3).transpose(2, 0, 1)
    # take sample uvd point every 10 step for each H, W direction
    sampleuvd = uvd[:, ::10, ::10]
    samplexyz = xyz[:, ::10, ::10]
    ax1.imshow(img)
    ax2.scatter(*sampleuvd)
    ax3.scatter(*samplexyz)
    ax2.view_init(-60, -90)
    ax3.view_init(-60, 90)
    ax3.invert_xaxis()
    # ax2.set_aspect("equal")
    # ax3.set_aspect("equal")
    fig.tight_layout()


interact(visualize_uvd_xyz, k=[1, 2, 3], f=range(2, 100+1))

# +
DEFAULT_NUM_KEYPOINTS = 36
DEFAULT_KEYPOINT_NAMES = [
    "F1_KNU3_A",
    "F1_KNU3_B",
    "F1_KNU2_A",
    "F1_KNU2_B",
    "F1_KNU1_A",
    "F1_KNU1_B",
    "F2_KNU3_A",
    "F2_KNU3_B",
    "F2_KNU2_A",
    "F2_KNU2_B",
    "F2_KNU1_A",
    "F2_KNU1_B",
    "F3_KNU3_A",
    "F3_KNU3_B",
    "F3_KNU2_A",
    "F3_KNU2_B",
    "F3_KNU1_A",
    "F3_KNU1_B",
    "F4_KNU3_A",
    "F4_KNU3_B",
    "F4_KNU2_A",
    "F4_KNU2_B",
    "F4_KNU1_A",
    "F4_KNU1_B",
    "TH_KNU3_A",
    "TH_KNU3_B",
    "TH_KNU2_A",
    "TH_KNU2_B",
    "TH_KNU1_A",
    "TH_KNU1_B",
    "PALM_1",
    "PALM_2",
    "PALM_3",
    "PALM_4",
    "PALM_5",
    "PALM_6",
]

NUM_KEYPOINT_NAMES=21
KEYPOINT_NAMES = [
    "little_tip",
    "little_dip",
    "little_pip",
    "little_mcp",
    "ring_tip",
    "ring_dip",
    "ring_pip",
    "ring_mcp",
    "middle_tip",
    "middle_dip",
    "middle_pip",
    "middle_mcp",
    "index_tip",
    "index_dip",
    "index_pip",
    "index_mcp",
    "thumb_tip",
    "thumb_dip",
    "thumb_pip",
    "thumb_mcp",
    "wrist",
]


CONVERTER={
    "wrist":"PALM_6",
}

for f,fID in zip(["thumb","index","middle","ring","little"],["TH","F4","F3","F2","F1"]):
    for p,pID in zip(["tip","dip","pip","mcp"],["KNU3_A","KNU3_B","KNU2_B","KNU1_A"]):
        key="_".join([f,p])
        value="_".join([fID,pID])
        CONVERTER[key]=value

assert DEFAULT_KEYPOINT_NAMES== list(mat["joint_names"])
assert len(DEFAULT_KEYPOINT_NAMES) == DEFAULT_NUM_KEYPOINTS
assert len(KEYPOINT_NAMES)==NUM_KEYPOINT_NAMES

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
EDGE_NAMES = []

for f in ["index", "middle", "ring", "little", "thumb"]:
    for p, q in pairwise(["wrist", "mcp", "pip", "dip", "tip"]):
        color = BASE_COLOR[f]
        if p != "wrist":
            p = "_".join([f, p])
        q = "_".join([f, q])
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
print(EDGE_NAMES)

# +
# %matplotlib inline

# These parameters are taken from convert_xyz_to_uvd.m

u0 = 640/2
v0 = 480/2
fx = 588.036865
fy = 587.075073

camera_intr = np.array([
    [fx, 0., u0],
    [0., fy, v0],
    [0., 0., 1.]
])



def overlay_pts_into_depth_map(k, f,kp):
    fig = plt.figure(figsize=(8, 10))
    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413, projection="3d")
    ax4 = fig.add_subplot(414, projection="3d")
    label_xyz3d(ax3)
    label_xyz3d(ax4)

    fmt = os.path.join(dataset_dir, "dataset", "train", "{0}_{1}_{2:07d}.png")
    rgb_file = fmt.format("rgb", k, f)
    depth_file = fmt.format("depth", k, f)
    synthdepth_file = fmt.format("synthdepth", k, f)
    print(rgb_file)
    img = imageio.imread(rgb_file)
    depth = decode_depthimg(imageio.imread(depth_file))
    synthdepth = imageio.imread(synthdepth_file)
    topbit = synthdepth[:, :, 1]
    lowerbit = synthdepth[:, :, 2]
    synthdepth = 2**8 * topbit + lowerbit
    uvd = convert_depth_to_uvd(depth)

    us, vs, ds = mat["joint_uvd"][k-1][f].transpose()
    ax1.imshow(img)
    ax2.imshow(synthdepth)
    #ax1.scatter(us, vs)
    ax3.scatter(us, vs, ds, color="red")
    joint = mat["joint_xyz"][k-1][f].copy()
    print(joint.shape)    
    # flip v-direction for our purpose
    joint[:, 1] = -joint[:, 1]

    if kp=="all":
        joint2D_hom = joint.dot(camera_intr.transpose())
        joint2D_hom = joint2D_hom/joint2D_hom[:, 2:]
        joint2D = joint2D_hom[:, :2]
        ax1.scatter(*joint2D.transpose())
    elif kp=="21":
        target_indices = [DEFAULT_KEYPOINT_NAMES.index(CONVERTER[k]) for k in KEYPOINT_NAMES]
        joint = joint[target_indices]
        joint2D_hom = joint.dot(camera_intr.transpose())
        joint2D_hom = joint2D_hom/joint2D_hom[:, 2:]
        joint2D = joint2D_hom[:, :2]
        ax1.scatter(*joint2D.transpose())
        for s,t in EDGES:
            ax1.plot(*joint2D[[s,t]].transpose())
    else:
        joint2D_hom = joint.dot(camera_intr.transpose())
        joint2D_hom = joint2D_hom/joint2D_hom[:, 2:]
        joint2D = joint2D_hom[:, :2]
        idx=DEFAULT_KEYPOINT_NAMES.index(CONVERTER[kp])
        ax1.scatter(*joint2D[idx].transpose())
    ax4.scatter(*joint.transpose(), color="red")
    _, H, W = uvd.shape
    # fit interface
    # 3,H,W -> H,W,3 -> [N,3], where N=H*W
    xyz = convert_uvd_to_xyz(uvd.transpose(1, 2, 0).reshape(-1, 3))
    # restore shape
    # N,3 -> H,W,3 -> 3,H,W
    xyz = xyz.reshape(H, W, 3).transpose(2, 0, 1)
    # take sample uvd point every 10 step for each H, W direction
    sampleuvd = uvd[:, ::10, ::10]
    samplexyz = xyz[:, ::10, ::10]
    ax1.imshow(img)
    ax3.scatter(*sampleuvd,  alpha=0.5)
    ax4.scatter(*samplexyz, alpha=0.1)

    ax3.view_init(-65, -90)
    ax4.view_init(-65, 90)
    ax4.invert_xaxis()
    # ax3.set_aspect("equal")
    # ax4.set_aspect("equal")
    fig.tight_layout()


kps =["21","all"]+ KEYPOINT_NAMES
interact(overlay_pts_into_depth_map, k=[1, 2, 3], f=range(2, 100+1), kp=kps)
