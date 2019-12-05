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

# # Apply Augmentation

# +
import configparser
import os

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
import chainercv
from chainercv import transforms

from ipywidgets import interact

import sys
sys.path.append("../../src/")
# -

from pose.visualizations import vis_image, vis_point, vis_edge, vis_pose
from pose.hand_dataset.selector import select_dataset
from pose.hand_dataset.common_dataset import COLOR_MAP, STANDARD_KEYPOINT_NAMES, EDGES

# define constants
KEYPOINT_NAMES = STANDARD_KEYPOINT_NAMES
POINT_COLOR = [COLOR_MAP[k] for k in KEYPOINT_NAMES]
EDGE_COLOR = [COLOR_MAP[s, t] for s, t in EDGES]

# # visualize raw dataset

# +
config = configparser.ConfigParser()
config.read("../../src/config_pose.ini")
# force to set
config["dataset"]["train_set"]="fhad"
config["dataset"]["val_set"]="fhad"
config["dataset"]["use_rgb"]="yes"
config["dataset"]["use_depth"]="yes"

# Uh... ugly...
concatenated_dataset=select_dataset(config, return_data=["train_set"], debug=True)
dataset=concatenated_dataset._datasets[0].base


# +
def visualize_dataset(idx):
    example = dataset.get_example(idx)
    print(example.keys())
    rgb_joint_zyx = example["rgb_joint"]
    depth_joint_zyx = example["depth_joint"]
    rgb = chainercv.utils.read_image(example["rgb_path"])
    depth = dataset.read_depth(example["depth_path"])
    rgb_vu = dataset.rgb_camera.zyx2vu(rgb_joint_zyx)
    depth_vu = dataset.depth_camera.zyx2vu(depth_joint_zyx)

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223, projection="3d")
    ax4 = fig.add_subplot(224, projection="3d")

    vis_pose(depth_vu, EDGES, img=depth, edge_color=EDGE_COLOR,
             point_color=POINT_COLOR, ax=ax1)
    vis_pose(rgb_vu, EDGES, img=rgb, edge_color=EDGE_COLOR,
             point_color=POINT_COLOR, ax=ax2)

    vis_pose(depth_joint_zyx, indices=EDGES, edge_color=EDGE_COLOR,
             point_color=POINT_COLOR, ax=ax3)
    vis_pose(rgb_joint_zyx, indices=EDGES, edge_color=EDGE_COLOR,
             point_color=POINT_COLOR, ax=ax4)
    for ax in [ax3, ax4]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(-65, -90)


sample_indices = np.random.choice(range(len(dataset)), 100)
interact(visualize_dataset, idx=sample_indices)


# -

# # Flip

# +
def flip_point_xyz(camera, xyz, size, x_flip=False, y_flip=False):
    uv, z = camera.xyz2uv(xyz, return_z=True)
    vu = uv[:, ::-1]
    W, H = size
    flipped_vu = transforms.flip_point(
        vu[np.newaxis],
        (H, W),
        x_flip=x_flip,
        y_flip=y_flip
    )
    flipped_uv = np.squeeze(flipped_vu)[:, ::-1]
    flipped_xyz = camera.uv2xyz(flipped_uv, z)
    return flipped_xyz


def flip_point_zyx(camera, zyx, size, x_flip=False, y_flip=False):
    print(zyx.shape)
    vu, z = camera.zyx2vu(zyx, return_z=True)
    H, W = size
    flipped_vu = transforms.flip_point(
        vu[np.newaxis],
        (H, W),
        x_flip=x_flip,
        y_flip=y_flip
    )
    flipped_vu = np.squeeze(flipped_vu)
    flipped_zyx = camera.vu2zyx(np.squeeze(flipped_vu), z)
    return flipped_zyx


# -

def flip(image, zyx, vu, camera, x_flip=False, y_flip=False):
    C, H, W = image.shape
    zyx_flipped = flip_point_zyx(
        camera, zyx, (H, W), x_flip=x_flip, y_flip=y_flip)
    image_flipped = transforms.flip(image, x_flip=x_flip, y_flip=y_flip)
    vu_flipped = transforms.flip_point(
        vu,
        (H, W),
        x_flip=x_flip,
        y_flip=y_flip,
    )
    return image_flipped, zyx_flipped, vu_flipped


# +
def visualize_flip(idx, y_flip=False, x_flip=False):
    example = dataset.get_example(idx)
    rgb_joint_zyx = example["rgb_joint"]
    depth_joint_zyx = example["depth_joint"]
    rgb = chainercv.utils.read_image(example["rgb_path"])
    depth = dataset.read_depth(example["depth_path"])
    rgb_vu = dataset.rgb_camera.zyx2vu(rgb_joint_zyx)
    depth_vu = dataset.depth_camera.zyx2vu(depth_joint_zyx)
    rgb_vu = np.expand_dims(rgb_vu, axis=0)
    depth_vu = np.expand_dims(depth_vu, axis=0)

    depth_flipped, depth_joint_zyx_flipped, depth_vu_flipped = flip(
        depth,
        depth_joint_zyx,
        depth_vu,
        dataset.depth_camera,
        x_flip=x_flip,
        y_flip=y_flip,
    )

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223, projection="3d")
    ax4 = fig.add_subplot(224, projection="3d")

    vis_pose(depth_vu, EDGES, img=depth, edge_color=EDGE_COLOR,
             point_color=POINT_COLOR, ax=ax1)
    debug_vu = np.expand_dims(dataset.depth_camera.zyx2vu(
        depth_joint_zyx_flipped), axis=0)
    vis_pose(debug_vu, EDGES, img=depth_flipped, edge_color=EDGE_COLOR,
             point_color=POINT_COLOR, ax=ax2)
    # plot 3D
    vis_pose(depth_joint_zyx, indices=EDGES, edge_color=EDGE_COLOR,
             point_color=POINT_COLOR, ax=ax3)
    vis_pose(depth_joint_zyx_flipped, indices=EDGES, edge_color=EDGE_COLOR,
             point_color=POINT_COLOR, ax=ax4)
    for ax in [ax3, ax4]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(-65, -90)


interact(visualize_flip, idx=sample_indices)
# -

# # Rotate

# +
from math import sin, cos


def rotate_point_uv(uv, angle, center_uv):
    ndim = uv.ndim
    if ndim == 3:
        uv = uv.squeeze()
    c_u, c_v = center_uv
    theta = np.deg2rad(angle)
    rmat_uv = np.array([
        [cos(theta), -sin(theta)],
        [sin(theta),  cos(theta)],
    ], dtype=uv.dtype).transpose()
    uv = uv-center_uv
    rot_uv = uv @ rmat_uv
    rot_uv = rot_uv+center_uv
    if ndim == 3:
        rot_uv = np.expand_dims(rot_uv, axis=0)
    return rot_uv


def rotate_point_xyz(camera, xyz, angle, center_uv):
    uv, z = camera.xyz2uv(xyz, return_z=True)
    rot_uv = rotate_point_uv(uv, angle, center_uv)
    xyz = camera.uv2xyz(rot_uv, z)
    return xyz


def rotate_point_vu(vu, angle, center_vu):
    ndim = vu.ndim
    if ndim == 3:
        vu = vu.squeeze()
    c_v, c_u = center_vu
    theta = np.deg2rad(angle)
    P = np.array([
        [0, 1],
        [1, 0],
    ], dtype=vu.dtype).transpose()
    rmat = np.array([
        [cos(theta), -sin(theta)],
        [sin(theta),  cos(theta)],
    ], dtype=vu.dtype).transpose()
    rmat_vu = P @ rmat @ P
    vu = vu-center_vu
    rot_vu = vu @ rmat_vu
    rot_vu = rot_vu+center_vu
    if ndim == 3:
        rot_vu = np.expand_dims(rot_vu, axis=0)
    return rot_vu


def rotate_point_zyx(camera, zyx, angle, center_vu):
    vu, z = camera.zyx2vu(zyx, return_z=True)
    rot_vu = rotate_point_vu(vu, angle, center_vu)
    zyx = camera.vu2zyx(rot_vu, z)
    return zyx


# -

def rotate(image, zyx, vu, angle, camera):
    C, H, W = image.shape
    center_vu = (H/2, W/2)
    # to make compatibility between transforms and rot_point_vu
    image_angle = angle
    point_angle = -angle
    zyx_rot = rotate_point_zyx(camera, zyx, point_angle, center_vu)
    image_rot = transforms.rotate(image, image_angle, expand=False)
    vu_rot = rotate_point_vu(vu, point_angle, center_vu)
    return image_rot, zyx_rot, vu_rot


# +
def visualize_rotate(idx, angle, vis_vu=False):
    example = dataset.get_example(idx)
    rgb_joint_zyx = example["rgb_joint"]
    depth_joint_zyx = example["depth_joint"]
    rgb = chainercv.utils.read_image(example["rgb_path"])
    depth = dataset.read_depth(example["depth_path"])
    rgb_vu = dataset.rgb_camera.zyx2vu(rgb_joint_zyx)
    depth_vu = dataset.depth_camera.zyx2vu(depth_joint_zyx)
    rgb_vu = np.expand_dims(rgb_vu, axis=0)
    depth_vu = np.expand_dims(depth_vu, axis=0)
    depth_rot, depth_joint_zyx_rot, depth_vu_rot = rotate(
        depth,
        depth_joint_zyx,
        depth_vu,
        angle,
        dataset.depth_camera,
    )

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223, projection="3d")
    ax4 = fig.add_subplot(224, projection="3d")

    vis_pose(depth_vu, EDGES, img=depth, edge_color=EDGE_COLOR,
             point_color=POINT_COLOR, ax=ax1)

    if vis_vu:
        vis_pose(depth_vu_rot, EDGES, img=depth_rot, edge_color=EDGE_COLOR,
                 point_color=POINT_COLOR, ax=ax2)
    else:
        debug_vu = np.expand_dims(
            dataset.depth_camera.zyx2vu(depth_joint_zyx_rot), axis=0)
        vis_pose(debug_vu, EDGES, img=depth_rot, edge_color=EDGE_COLOR,
                 point_color=POINT_COLOR, ax=ax2)
    # plot 3D
    vis_pose(depth_joint_zyx, indices=EDGES, edge_color=EDGE_COLOR,
             point_color=POINT_COLOR, ax=ax3)
    vis_pose(depth_joint_zyx_rot, indices=EDGES, edge_color=EDGE_COLOR,
             point_color=POINT_COLOR, ax=ax4)
    for ax in [ax3, ax4]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(-65, -90)


interact(visualize_rotate, idx=sample_indices, angle=range(-180, 181, 30))
# -

# # Crop around hand

# +
crop3dH, crop3dW = 150, 150
crop2dH, crop2dW = 224, 224
crop3dD = 150


def crop_domain(image, domain, fill=0):
    """
    image.shape should be (C,H,W)
    The order of domain should be [ymin,xmin,ymax,xmax]
    """
    # m: min, M:max
    ym, xm, yM, xM = domain
    # Select domain where to clip
    C, H, W = image.shape
    # s: select
    sxm = max(0, xm)
    sxM = min(W, xM)
    sym = max(0, ym)
    syM = min(H, yM)
    outH, outW = yM - ym, xM - xm
    canvas = np.empty((C, outH, outW), dtype=image.dtype)
    canvas[:] = np.array(fill).reshape((-1, 1, 1))
    # where to Put clipped image on canvas
    # p: put
    pym = max(0, sym - ym)
    pxm = max(0, sxm - xm)
    pyM = min(outH, syM - ym)
    pxM = min(outW, sxM - xm)
    if pym == pyM:
        print(H, W)
        print(ym, xm, yM, xM)
        print(pym, pxm, pyM, pxM)
        raise Exception
    # TODO:express as slice
    canvas[:, pym:pyM, pxm:pxM] = image[:, sym:syM, sxm:sxM]
    param = {}
    param['y_offset'] = -sym + pym
    param['x_offset'] = -sxm + pxm

    return canvas, param


def calc_com(pts, z=None):
    """
    calculate center of mass for given points pts
    """
    if z is None:
        return np.mean(pts, axis=0)
    return np.mean(pts, axis=0), np.mean(z, axis=0)


# -

# # compose affine

from pose.graphics.camera import CameraIntr
from pose.hand_dataset.image_utils import convert_depth_to_uvd

# +
import sympy
symbols = []
for i in range(3):
    for j in range(3):
        s = "x_{}{}".format(i, j)
        symbols.append(s)
symbols = ' '.join(symbols)
m = sympy.symbols(symbols)
m = np.array(m).reshape(3, 3)
P = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
])

m_zyx = P.dot(m.dot(P))
m_xyz = P.dot(m_zyx.dot(P))
m_xyz
# -

# !pip install sympy
import sympy

u0, v0 = sympy.symbols("u0 v0")
u, v = sympy.symbols("u v")
sk = sympy.symbols("sk")
fx, fy = sympy.symbols("fx fy")
sx, sy = sympy.symbols("sx sy")

# +
c = np.array([
    [1, sk, u0],
    [0, 1, v0],
    [0, 0, 1],
])

t = np.array([
    [1, 0, u],
    [0, 1, v],
    [0, 0, 1],
])

s = np.array([
    [sx, 0, 0],
    [0, sy, 0],
    [0, 0, 1],
])

t.dot(c), s.dot(c)


# -

def crop(image, joint_zyx, camera, return_param=False):
    vu, z = camera.zyx2vu(joint_zyx, return_z=True)

    vu_com, z_com = calc_com(vu, z)
    zyx_com = camera.vu2zyx(
        vu_com[np.newaxis],
        z_com[np.newaxis]
    ).squeeze()
    z_com, y_com, x_com = zyx_com
    xmin = x_com-crop3dW / 2
    ymin = y_com-crop3dH / 2
    xmax = xmin+crop3dW
    ymax = ymin+crop3dH
    [
        [vmin, umin],
        [vmax, umax],
    ] = camera.zyx2vu(np.array([
        [z_com, ymin, xmin],
        [z_com, ymax, xmax],
    ])).astype(int)
    domain = [vmin, umin, vmax, umax]

    cropped, crop_param = crop_domain(image, domain)

    translated = camera.translate_camera(
        y_offset=crop_param["y_offset"],
        x_offset=crop_param["x_offset"]
    )

    vu_cropped = translated.zyx2vu(joint_zyx)

    if return_param:
        param = dict()
        param["zyx_com"] = zyx_com
        param["z_com"] = z_com
        param["y_com"] = y_com
        param["x_com"] = x_com
        return cropped, vu_cropped, translated, param
    else:
        return cropped, vu_cropped, translated


# +
import copy
# %matplotlib notebook


def visualize_crop(i):
    example = dataset.get_example(i)
    rgb_joint_zyx = example["rgb_joint"]
    depth_joint_zyx = example["depth_joint"]
    rgb = chainercv.utils.read_image(example["rgb_path"])
    depth = dataset.read_depth(example["depth_path"])

    depth_cropped, depth_vu_cropped, depth_camera_cropped, depth_crop_param = crop(
        depth, depth_joint_zyx, dataset.depth_camera, return_param=True)

    rgb_cropped, rgb_vu_cropped, rgb_camera_cropped, rgb_crop_param = crop(
        rgb, rgb_joint_zyx, dataset.rgb_camera, return_param=True)

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223, projection="3d")
    ax4 = fig.add_subplot(224, projection="3d")

    vis_image(depth_cropped, ax1)
    vis_pose(depth_vu_cropped, EDGES, point_color=POINT_COLOR,
             edge_color=EDGE_COLOR, ax=ax1)

    vis_image(rgb_cropped, ax2)
    vis_pose(rgb_vu_cropped, EDGES, point_color=POINT_COLOR,
             edge_color=EDGE_COLOR, ax=ax2)

    # plot 3D
    # pull back depth map
    uvd = convert_depth_to_uvd(depth_cropped)
    u, v, d = uvd[:, ::10, ::10]
    u = u.reshape(-1, 1)
    v = v.reshape(-1, 1)
    z = d.reshape(-1, 1)
    vu = np.concatenate([v, u], axis=1)
    zyx = depth_camera_cropped.vu2zyx(vu, z)
    vis_point(zyx, ax=ax3)
    zyx_com = depth_crop_param["zyx_com"]
    vis_pose(depth_joint_zyx, indices=EDGES, edge_color=EDGE_COLOR,
             point_color=POINT_COLOR, ax=ax3)
    vis_pose(rgb_joint_zyx, indices=EDGES, edge_color=EDGE_COLOR,
             point_color=POINT_COLOR, ax=ax4)
    for ax in [ax3, ax4]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(-65, -90)


interact(visualize_crop, i=sample_indices)


# -

# # resize

def scale(image, joint_zyx, camera, size, fit_short=True):
    _, inH, inW = image.shape
    out_image = chainercv.transforms.scale(
        image,
        size=max(crop2dH, crop2dW),
        fit_short=fit_short,
    )

    _, outH, outW = out_image.shape

    y_scale = float(outH)/inH
    x_scale = float(outW)/inW
    camera_scaled = camera.scale_camera(y_scale=y_scale, x_scale=x_scale)
    vu = camera_scaled.zyx2vu(joint_zyx)
    return out_image, vu, camera_scaled


# +
import copy
# %matplotlib notebook


def visualize_scale(i):
    example = dataset.get_example(i)
    rgb_joint_zyx = example["rgb_joint"]
    depth_joint_zyx = example["depth_joint"]
    rgb = chainercv.utils.read_image(example["rgb_path"])
    depth = dataset.read_depth(example["depth_path"])

    depth_scaled, depth_vu_scaled, depth_camera_scaled = scale(
        depth, depth_joint_zyx, dataset.depth_camera, size=max(crop2dH, crop2dW))

    rgb_scaled, rgb_vu_scaled, rgb_camera_scaled = scale(
        rgb, rgb_joint_zyx, dataset.rgb_camera, size=max(crop2dH, crop2dW))

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223, projection="3d")
    ax4 = fig.add_subplot(224, projection="3d")

    vis_image(depth_scaled, ax1)
    vis_pose(depth_vu_scaled, EDGES, point_color=POINT_COLOR,
             edge_color=EDGE_COLOR, ax=ax1)

    vis_image(rgb_scaled, ax2)
    vis_pose(rgb_vu_scaled, EDGES, point_color=POINT_COLOR,
             edge_color=EDGE_COLOR, ax=ax2)

    # plot 3D
    # pull back depth map
    uvd = convert_depth_to_uvd(depth_scaled)
    u, v, d = uvd[:, ::10, ::10]
    u = u.reshape(-1, 1)
    v = v.reshape(-1, 1)
    z = d.reshape(-1, 1)
    uv = np.concatenate([u, v], axis=1)
    xyz = depth_camera_scaled.uv2xyz(uv, z)

    ax3.scatter(*xyz.transpose(), alpha=0.4)
    vis_pose(depth_joint_zyx, indices=EDGES, edge_color=EDGE_COLOR,
             point_color=POINT_COLOR, ax=ax3)
    vis_pose(rgb_joint_zyx, indices=EDGES, edge_color=EDGE_COLOR,
             point_color=POINT_COLOR, ax=ax4)
    for ax in [ax3, ax4]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(-65, -90)


interact(visualize_scale, i=sample_indices)


# -

def resize_contain(image, joint_zyx, camera, size, fill=0, return_param=False):
    _, inH, inW = image.shape
    resized, resize_param = transforms.resize_contain(
        image,
        size=size,
        return_param=True,
        fill=fill,
    )
    y_scale,x_scale=resize_param["scaled_size"]/np.array([inH,inW])

    print(resize_param)
    vu=camera.zyx2vu(joint_zyx.copy())
    vu=np.expand_dims(vu,axis=0)
    vu = transforms.resize_point(
        vu,
        in_size=(inH, inW),
        out_size=resize_param["scaled_size"]
    )
    vu = transforms.translate_point(
        vu,
        y_offset=resize_param["y_offset"],
        x_offset=resize_param["x_offset"]
    )
    
    camera_scaled = camera.scale_camera(y_scale=y_scale, x_scale=x_scale)
    camera_resized = camera_scaled.translate_camera(
        y_offset=resize_param["y_offset"], 
        x_offset=resize_param["x_offset"]
    )
    vu = camera_resized.zyx2vu(joint_zyx)
    return resized, vu, camera_resized


# +
import copy
# %matplotlib notebook

crop2dH,crop2dW=300,300
def visualize_resize_contain(i):
    example = dataset.get_example(i)
    rgb_joint_zyx = example["rgb_joint"]
    depth_joint_zyx = example["depth_joint"]
    rgb = chainercv.utils.read_image(example["rgb_path"])
    depth = dataset.read_depth(example["depth_path"])

    depth_resized, depth_vu_resized, depth_camera_resized = resize_contain(
        depth, depth_joint_zyx, dataset.depth_camera, size=(crop2dH, crop2dW))

    rgb_resized, rgb_vu_resized, rgb_camera_resized = resize_contain(
        rgb, rgb_joint_zyx, dataset.rgb_camera, size=(crop2dH, crop2dW))

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223, projection="3d")
    ax4 = fig.add_subplot(224, projection="3d")

    vis_image(depth_resized, ax1)
    vis_pose(depth_vu_resized, EDGES, point_color=POINT_COLOR,
             edge_color=EDGE_COLOR, ax=ax1)

    vis_image(rgb_resized, ax2)
    vis_pose(rgb_vu_resized, EDGES, point_color=POINT_COLOR,
             edge_color=EDGE_COLOR, ax=ax2)

    # plot 3D
    # pull back depth map
    uvd = convert_depth_to_uvd(depth_resized)
    u, v, d = uvd[:, ::10, ::10]
    u = u.reshape(-1, 1)
    v = v.reshape(-1, 1)
    z = d.reshape(-1, 1)
    uv = np.concatenate([u, v], axis=1)
    xyz = depth_camera_resized.uv2xyz(uv, z)

    ax3.scatter(*xyz.transpose(), alpha=0.4)
    vis_pose(depth_joint_zyx, indices=EDGES, edge_color=EDGE_COLOR,
             point_color=POINT_COLOR, ax=ax3)
    vis_pose(rgb_joint_zyx, indices=EDGES, edge_color=EDGE_COLOR,
             point_color=POINT_COLOR, ax=ax4)
    for ax in [ax3, ax4]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(-65, -90)


interact(visualize_resize_contain, i=sample_indices)
# -


