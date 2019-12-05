# -*- coding: utf-8 -*-
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

# # crop 3D bounding box for first_person_action_cvpr2018

# + {"language": "javascript"}
# // disable auto scrolling
# IPython.OutputArea.prototype._should_scroll = function(lines) {
#     return false
# }
#

# +
from itertools import tee
import os

import chainercv
from chainercv import visualizations

import imageio
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ipywidgets import interact
# -

dataset_dir = os.path.expanduser(
    "~/dataset/fhad")


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


# +
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

# +
NUM_KEYPOINTS = 21
# ’T’, ’I’, ’M’, ’R’, ’P’ denote ’Thumb’, ’Index’, ’Middle’, ’Ring’, ’Pinky’ fingers
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

assert len(KEYPOINT_NAMES) == NUM_KEYPOINTS
# (R,G,B)
BASE_COLOR = {
    "I": (0, 0, 255),
    "M": (0, 255, 0),
    "R": (255, 255, 0),
    "P": (255, 0, 0),
    "T": (255, 0, 255),
    "Wrist": (50, 50, 50),
}

COLOR_MAP = {"Wrist": BASE_COLOR["Wrist"]}

EDGE_NAMES = []
for f in ["T", "I", "M", "R", "P"]:
    for s, t in pairwise(["Wrist", "MCP", "PIP", "DIP", "TIP"]):
        color = BASE_COLOR[f]
        if s == "Wrist":
            t = f+t
        else:
            s = f+s
            t = f+t
        EDGE_NAMES.append([s, t])
        COLOR_MAP[s, t] = color
        COLOR_MAP[t] = color

EDGES = [[KEYPOINT_NAMES.index(s), KEYPOINT_NAMES.index(t)]
         for s, t in EDGE_NAMES]

for s, t in EDGE_NAMES:
    COLOR_MAP[
        KEYPOINT_NAMES.index(s),
        KEYPOINT_NAMES.index(t)
    ] = COLOR_MAP[s, t]
    COLOR_MAP[KEYPOINT_NAMES.index(s)] = COLOR_MAP[s]
    COLOR_MAP[KEYPOINT_NAMES.index(t)] = COLOR_MAP[t]
# convert value as np.array
COLOR_MAP = {k: np.array(v) for k, v in COLOR_MAP.items()}

# +
"""
camera parameter for image sensor
"""
# Image center
u0 = 935.732544
v0 = 540.681030
# Focal Length
fx = 1395.749023
fy = 1395.749268

dtype = np.float32

cam_intr = np.array([
    [fx, 0, u0],
    [0, fy, v0],
    [0, 0, 1],
], dtype=dtype).transpose()

cam_extr = np.array([
    [0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
    [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
    [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
    [0, 0, 0, 1],
], dtype=dtype).transpose()

"""
camera parameter for depth sensor
"""
# Image center:
depth_u0 = 315.944855
depth_v0 = 245.287079
# Focal Length:
depth_fx = 475.065948
depth_fy = 475.065857

depth_cam_intr = np.array([
    [depth_fx, 0, depth_u0],
    [0, depth_fy, depth_v0],
    [0, 0, 1],
]).transpose()



# +
cam_intr_xyz = cam_intr
P=np.array([
    [0,0,1],
    [0,1,0],
    [1,0,0]
],dtype=cam_intr_xyz.dtype)

depth_cam_intr_xyz=depth_cam_intr

cam_intr_zyx= P @ cam_intr_xyz @ P
depth_cam_intr_zyx= P @ depth_cam_intr_xyz @ P


pull_back_xyz = np.array([
    [1/fx, 0, -u0/fx],
    [0, 1/fy, -v0/fy],
    [0, 0, 1],
], dtype=dtype).transpose()

pull_back_zyx = P @ pull_back_xyz @ P

pull_back_depth_xyz = np.array([
    [1/depth_fx, 0, -depth_u0/depth_fx],
    [0, 1/depth_fy, -depth_v0/depth_fy],
    [0, 0, 1],
], dtype=dtype).transpose()

pull_back_depth_zyx = P @ pull_back_depth_xyz @ P

def xyz2uv(xyz, return_z=False):
    z = xyz[:, 2:]
    uv_ = xyz/z @ cam_intr_xyz
    uv = uv_[:, :2]
    if return_z:
        return uv, z
    return uv

def zyx2vu(zyx, return_z=False):
    z = zyx[:, :1]
    zvu = zyx/z @ cam_intr_zyx
    vu = zvu[:, 1:]
    if return_z:
        return vu,z
    return vu

def uv2xyz(uv, z):
    nk, *_ = uv.shape
    hom_uv = np.concatenate([uv, np.ones((nk, 1), dtype=dtype)], axis=1)
    xy_ = hom_uv @ pull_back_xyz
    xyz = z*xy_
    return xyz

def vu2zyx(vu, z):
    nk, *_ = vu.shape
    hom_vu = np.concatenate([np.ones((nk, 1), dtype=dtype), vu], axis=1)
    _yx = hom_vu @ pull_back_zyx
    zyx = z*_yx
    return zyx


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

def zyx2depth_vu(zyx, return_z=False):
    z = zyx[:, :1]
    zvu = zyx/z @ depth_cam_intr_zyx
    vu = zvu[:, 1:]
    if return_z:
        return vu,z
    return vu


def xyz2depth_uv(xyz, return_z=False):
    z = xyz[:, 2:]
    uv_ = xyz/z @ depth_cam_intr_xyz
    uv = uv_[:, :2]
    if return_z:
        return uv, z
    return uv


def depth_vu2zyx(vu, z):
    nk, *_ = vu.shape
    hom_vu = np.concatenate([np.ones((nk, 1), dtype=dtype), vu], axis=1)
    _yx = hom_vu @ pull_back_depth_zyx
    zyx = z*_yx
    return zyx


# -

def world_xyz2xyz(world_xyz):
    nk, *_ = world_xyz.shape
    hom_world_xyz = np.concatenate([world_xyz, np.ones((nk, 1))], axis=1)
    hom_cam_xyz = hom_world_xyz @ cam_extr
    cam_xyz = hom_cam_xyz[:, :3]
    return cam_xyz

# +
# %matplotlib notebook


def get_example(subject_id, action, seq_idx, frame_id):
    video_dir = os.path.join(dataset_dir, "Video_files")
    subject_dir = os.path.join(video_dir, "Subject_{}".format(subject_id))
    img_path = os.path.join(
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
    annotation_dir = os.path.join(dataset_dir, "Hand_pose_annotation_v1")
    skeleton_path = os.path.join(
        annotation_dir,
        "Subject_{}".format(subject_id),
        action,
        str(seq_idx),
        "skeleton.txt"
    )
    img = chainercv.utils.read_image(img_path)
    depth = np.expand_dims(imageio.imread(depth_path),axis=0)
    annotations = np.loadtxt(skeleton_path)
    world_joints = annotations[frame_id][1:].reshape(-1, 3)
    annot_id = annotations[frame_id][0]
    assert annot_id == frame_id
    example = {}
    example["image"] = img
    example["depth"] = depth
    example["world_joints"] = world_joints
    example["cam_joints"] = world_xyz2xyz(world_joints)
    return example


def visualize_dataseet(subject_id, action, seq_idx, frame_id):
    example = get_example(subject_id, action, seq_idx, frame_id)
    world_joints = example["world_joints"]
    img = example["image"]
    cam_joints = world_xyz2xyz(world_joints)
    fig = plt.figure(figsize=(5, 8))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313, projection="3d")
    chainercv.visualizations.vis_image(img,ax=ax1)
    ax2.imshow(example["depth"].squeeze())
    rgb_uv = xyz2uv(cam_joints).transpose()
    depth_uv = xyz2depth_uv(world_joints).transpose()
    colors = [COLOR_MAP[k]/255 for k in KEYPOINT_NAMES]
    ax1.scatter(*rgb_uv, color=colors)
    ax2.scatter(*depth_uv,color=colors)
    ax3.scatter(*cam_joints.transpose(), color=colors)
    label_3d(ax3)
    ax3.view_init(-65, -90)
    for s, t in EDGES:
        sx, sy, sz = cam_joints[s]
        tx, ty, tz = cam_joints[t]
        color = COLOR_MAP[s, t]/255
        ax3.plot([sx, tx], [sy, ty], [sz, tz], color=color)
        uv_st = xyz2uv(cam_joints[[s, t]])
        ax1.plot(*uv_st.transpose(), color=color)


interact(visualize_dataseet, subject_id=SUBJECT_INDICES,
         action=ACTION_LIST, seq_idx=SEQ_INDICES, frame_id=range(100))
# -

# # Crop uv image

# ## calc com=center of mass

# +
CENTER_KEYPOINT_NAME = "MMCP"
HAND_CENTER_IDX = KEYPOINT_NAMES.index(CENTER_KEYPOINT_NAME)

def get_hand_uvcenter(uv,z=None):
    if z is None:
        return uv[HAND_CENTER_IDX]
    return uv[HAND_CENTER_IDX], z[HAND_CENTER_IDX]


def calc_com(pts,z=None):
    """
    calculate center of mass for given points pts
    """
    if z is None:
        return np.mean(pts,axis=0)
    return np.mean(pts, axis=0),np.mean(z,axis=0)


# +

crop2dH, crop2dW = 368, 368


def crop_img(img, bbox):
    H, W = sizeHW
    Y, X = centerYX
    cropped = img[Y-H//2:Y+H//2, X-W//2:X+W//2]
    return cropped


def crop_domain(image, domain, fill=0):
    """
    image.shape should be (C,H,W)
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
    new_canvas = np.empty((C, outH, outW), dtype=image.dtype)
    new_canvas[:] = np.array(fill).reshape((-1, 1, 1))
    # where to Put clipped image on new_canvas
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
    new_canvas[:, pym:pyM, pxm:pxM] = image[:, sym:syM, sxm:sxM]
    param = {}
    param['y_offset'] = -sym + pym
    param['x_offset'] = -sxm + pxm

    return new_canvas, param


def crop_around_2d_center(subject_id, action, seq_idx, frame_id):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    example = get_example(subject_id, action, seq_idx, frame_id)
    cam_joints = example["cam_joints"]
    uv, z = xyz2uv(cam_joints, return_z=True)
    com_u, com_v = calc_com(uv)
    uv = np.round(uv).astype(np.int)
    umin, umax = uv[:, 0].min(), uv[:, 0].max()
    vmin, vmax = uv[:, 1].min(), uv[:, 1].max()
    u_ext = int(0.2*(vmax-vmin))
    v_ext = int(0.2*(umax-umin))
    domain = [vmin-v_ext, umin-u_ext, vmax+v_ext, umax+u_ext]
    img = example["image"]
    cropped, param = crop_domain(img, domain)
    chainercv.visualizations.vis_image(cropped, ax=ax1)
    cropped_point = np.array(
        [[[param["y_offset"]+com_v,
           param["x_offset"]+com_u]]]
    )
    uv = np.array([param["x_offset"], param["y_offset"]])+uv
    color = [COLOR_MAP[k]/255 for k in KEYPOINT_NAMES]
    ax1.scatter(*uv.transpose(), color=color)
    chainercv.visualizations.vis_point(cropped, cropped_point, ax=ax1)


interact(
    crop_around_2d_center,
    subject_id=SUBJECT_INDICES,
    action=ACTION_LIST,
    seq_idx=SEQ_INDICES,
    frame_id=range(100)
)
# -

# # crop around xyz which is pullback of "center of hand uv" or "MMCP"

# +

crop3dH, crop3dW, crop3dD = 192,192,192

def crop_around_3d_center(subject_id, action, seq_idx, frame_id):
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122,projection="3d")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.view_init(-90,-90)
    example = get_example(subject_id, action, seq_idx, frame_id)
    cam_joints = example["cam_joints"]
    uv, z_ = xyz2uv(cam_joints, return_z=True)
    uv_com,z_com = calc_com(uv,z_)
    xyz_com= uv2xyz(uv_com[np.newaxis], z_com[np.newaxis]).squeeze()
    x_com,y_com,z_com=xyz_com
    xmin,ymin,xmax,ymax=x_com-crop3dW/2,y_com-crop3dH/2,x_com+crop3dW/2,y_com+crop3dH/2
    print(xmin,ymin,xmax,ymax,z_com)
    [
         [umin,vmin],
         [umax,vmax],
    ]=xyz2uv(np.array([
        [xmin,ymin,z_com],
        [xmax,ymax,z_com],
    ])).astype(int)
    domain = [vmin, umin, vmax, umax]
    img = example["image"]
    cropped, param = crop_domain(img, domain)
    chainercv.visualizations.vis_image(cropped, ax=ax1)
    offset_uv=np.array([param["x_offset"], param["y_offset"]])
    uv = offset_uv+uv
    color = [COLOR_MAP[k]/255 for k in KEYPOINT_NAMES]
    ax1.scatter(*uv.transpose(), color=color)
    ax2.scatter(*(cam_joints-xyz_com).transpose(),color=color)
    for s, t in EDGES:
        joint_s_and_t = cam_joints[[s,t]]-xyz_com
        color = COLOR_MAP[s, t]/255
        uv_s_and_t=uv[[s,t]]
        ax1.plot(*uv_s_and_t.transpose(),color=color)
        ax2.plot(*joint_s_and_t.transpose(), color=color)

interact(
    crop_around_3d_center,
    subject_id=SUBJECT_INDICES,
    action=ACTION_LIST,
    seq_idx=SEQ_INDICES,
    frame_id=range(100)
)
# -

# # use zyx coordinate system

# +

crop3dH, crop3dW, crop3dD = 192,192,192

def crop_around_3d_center(subject_id, action, seq_idx, frame_id):
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122,projection="3d")
    label_3d(ax2)
    ax2.view_init(-90,-90)
    example = get_example(subject_id, action, seq_idx, frame_id)
    cam_joints = example["cam_joints"][:,::-1]
    vu, z_ = zyx2vu(cam_joints, return_z=True)
    vu_com,z_com = calc_com(vu,z_)
    zyx_com= vu2zyx(vu_com[np.newaxis], z_com[np.newaxis]).squeeze()
    z_com,y_com,x_com=zyx_com
    xmin,ymin,xmax,ymax=x_com-crop3dW/2,y_com-crop3dH/2,x_com+crop3dW/2,y_com+crop3dH/2
    print(xmin,ymin,xmax,ymax,z_com)
    [
         [vmin,umin],
         [vmax,umax],
    ]=zyx2vu(np.array([
        [z_com,ymin,xmin],
        [z_com,ymax,xmax],
    ])).astype(int)
    domain = [vmin, umin, vmax, umax]
    img = example["image"]
    cropped, param = crop_domain(img, domain)
    visualizations.vis_image(cropped, ax=ax1)
    offset_vu=np.array([param["y_offset"], param["x_offset"]])
    vu = offset_vu+vu
    color = [COLOR_MAP[k]/255 for k in KEYPOINT_NAMES]
    ax1.scatter(*vu[:,::-1].transpose(), color=color)

    ax2.scatter(*((cam_joints-zyx_com)[:,::-1]).transpose(),color=color)
    for s, t in EDGES:
        joint_s_and_t = cam_joints[[s,t]]-zyx_com
        color = COLOR_MAP[s, t]/255
        vu_s_and_t=vu[[s,t]]
        ax1.plot(*vu_s_and_t[:,::-1].transpose(),color=color)
        ax2.plot(*joint_s_and_t[:,::-1].transpose(), color=color)

interact(
    crop_around_3d_center,
    subject_id=SUBJECT_INDICES,
    action=ACTION_LIST,
    seq_idx=SEQ_INDICES,
    frame_id=range(100)
)

# +
import chainercv

def vis_image(img, ax):
    C,H,W = img.shape
    if C==1:
        # remove channnel dimension
        ax.imshow(img.squeeze())
    else:
        ax=chainercv.visualizations.vis_image(img,ax)
    return ax

def vis_point(point, img=None, color=None, ax=None):
    """Visualize points in an image.
    Taken From ChainerCV project
    """
    H, W = None, None
    if img is not None:
        ax = vis_image(img, ax=ax)
        _, H, W = img.shape
    n_inst = len(point)
    c = np.array(color)/255. if color is not None else None
    for i in range(n_inst):
        # note that the shape of `point[i]` is (K,N) and the format of one is (y, x), (z,y,x).
        # (K, N) -> (N, K)
        pts = point[i].transpose()  # (K,N) -> (N,K)
        # resort coordinate order : yx -> xy or zyx -> xyz
        pts = pts[::-1]
        ax.scatter(*pts, c=c)
    if W is not None:
        ax.set_xlim(left=0, right=W)
    if H is not None:
        ax.set_ylim(bottom=H - 1, top=0)
    return ax


def vis_edges(point, indices, color=None, ax=None):
    H, W = None, None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        print("IS NONE")
    n_inst = len(point)
    color = np.array(color)/255. if color is not None else [None]*len(indices)
    for i in range(n_inst):
        # note that the shape of `point[i]` is (K,N) and the format of one is (y, x) or (z,y,x).
        pts = point[i]
        for ((s, t), c) in zip(indices, color):
            # Select point which consists edge. It is a pair or point (start, target).
            # Note that [::-1] does resort coordinate order: yx -> xy or zyx -> xyz
            edge = pts[[s, t]].transpose()
            edge = edge[::-1]
            ax.plot(*edge, c=c)
    if W is not None:
        ax.set_xlim(left=0, right=W)
    if H is not None:
        ax.set_ylim(bottom=H - 1, top=0)
    return ax


# -

# # Resize Images

# +
from chainercv import transforms

crop3dH, crop3dW, crop3dD = 192, 192, 192
crop2dH, crop2dW = 224, 224


def crop_around_3d_center(subject_id, action, seq_idx, frame_id):
    global image
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection="3d")
    label_3d(ax2)
    ax2.view_init(-90, -90)
    example = get_example(subject_id, action, seq_idx, frame_id)
    cam_joints_zyx = example["cam_joints"][:, ::-1]
    vu, z_ = zyx2vu(cam_joints_zyx, return_z=True)
    vu_com, z_com = calc_com(vu, z_)
    zyx_com = vu2zyx(vu_com[np.newaxis], z_com[np.newaxis]).squeeze()
    z_com, y_com, x_com = zyx_com
    [
        xmin,
        ymin,
        xmax,
        ymax,
    ] = [
        x_com-crop3dW/2,
        y_com-crop3dH/2,
        x_com+crop3dW/2,
        y_com+crop3dH/2,
    ]
    [
        [vmin, umin],
        [vmax, umax],
    ] = zyx2vu(np.array([
        [z_com, ymin, xmin],
        [z_com, ymax, xmax],
    ])).astype(int)
    domain = [vmin, umin, vmax, umax]
    img = example["image"]

    cropped, crop_param = crop_domain(img, domain)
    offset_vu = np.array([crop_param["y_offset"], crop_param["x_offset"]])
    vu = np.expand_dims(vu, axis=0)
    vu = transforms.translate_point(
        vu,
        y_offset=crop_param["y_offset"],
        x_offset=crop_param["x_offset"]
    )
    _, inH, inW = cropped.shape
    resized, resize_param = transforms.resize_contain(
        cropped,
        size=(crop2dH, crop2dW),
        return_param=True
    )
    vu = transforms.resize_point(vu, in_size=(
        inH, inW), out_size=resize_param["scaled_size"])
    vu = transforms.translate_point(
        vu,
        y_offset=resize_param["y_offset"],
        x_offset=resize_param["x_offset"]
    )
    # visualize
    color = [COLOR_MAP[k] for k in KEYPOINT_NAMES]
    chainercv.visualizations.vis_image(resized, ax=ax1)
    vis_point(point=vu, ax=ax1, color=color)
    cropped_zyx = cam_joints_zyx-zyx_com
    vis_point(point=[cropped_zyx], ax=ax2, color=color)
    edge_color = [COLOR_MAP[s, t] for s, t in EDGES]
    vis_edges(point=vu, indices=EDGES, color=edge_color, ax=ax1)
    vis_edges(point=[cropped_zyx], indices=EDGES, color=edge_color, ax=ax2)


interact(
    crop_around_3d_center,
    subject_id=SUBJECT_INDICES,
    action=ACTION_LIST,
    seq_idx=SEQ_INDICES,
    frame_id=range(100)
)
# -

# # Crop depth
#

# +
from scipy import stats

VALID_MIN = 30
VALID_MAX = 2**(16-1)


def define_background(depth):
    valid_loc = np.logical_and(VALID_MIN <= depth, depth <= VALID_MAX)
    # define background as most frequently occurring number i.e. mode
    valid_depth = depth[valid_loc]
    mode_val, mode_num = stats.mode(valid_depth.ravel())
    background = mode_val.squeeze()
    return background


def normalize_depth(depth, z_com, z_size):
    z_half = z_size/2
    valid_loc = np.logical_and(VALID_MIN <= depth, depth <= VALID_MAX)
    invalid_loc = np.logical_not(valid_loc)
    depth[invalid_loc] = z_com+z_half
    # shift depth so that z_com goes to origin
    depth = depth-z_com
    # scale -1 to 1.
    # -1 is very close 1 is very far
    depth = depth/z_half
    # cut off
    depth[depth > 1] = 1.
    # cut off
    depth[depth < -1] = -1
    return depth


# -

# # Crop depth map and normalize it

# +
from chainercv import transforms

crop3dH, crop3dW = 150, 150
crop2dH, crop2dW = 224, 224
crop3dD = 150


def crop_around_3d_center(subject_id, action, seq_idx, frame_id):
    global image
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223, projection="3d")
    label_3d(ax3)
    ax3.view_init(-90, -90)
    example = get_example(subject_id, action, seq_idx, frame_id)
    joints_zyx = example["world_joints"][:, ::-1]
    vu, z_ = zyx2depth_vu(joints_zyx, return_z=True)
    vu_com, z_com = calc_com(vu, z_)
    zyx_com = depth_vu2zyx(vu_com[np.newaxis], z_com[np.newaxis]).squeeze()
    z_com, y_com, x_com = zyx_com
    [
        xmin,
        ymin,
        xmax,
        ymax,
    ] = [
        x_com-crop3dW/2,
        y_com-crop3dH/2,
        x_com+crop3dW/2,
        y_com+crop3dH/2,
    ]
    [
        [vmin, umin],
        [vmax, umax],
    ] = zyx2depth_vu(np.array([
        [z_com, ymin, xmin],
        [z_com, ymax, xmax],
    ])).astype(int)
    domain = [vmin, umin, vmax, umax]
    depth = example["depth"]
    cropped, crop_param = crop_domain(depth, domain)
    vu = np.expand_dims(vu, axis=0)
    vu = transforms.translate_point(
        vu,
        y_offset=crop_param["y_offset"],
        x_offset=crop_param["x_offset"]
    )
    _, inH, inW = cropped.shape

    if inH < crop2dH or inW < crop2dW:
        cropped = chainercv.transforms.scale(
            cropped, size=max(crop2dH, crop2dW), fit_short=True)
        vu = transforms.resize_point(
            vu,
            in_size=(inH, inW),
            out_size=cropped.shape[1:],
        )
        _, inH, inW = cropped.shape

    resized, resize_param = transforms.resize_contain(
        cropped,
        size=(crop2dH, crop2dW),
        return_param=True,
        fill=define_background(cropped),
    )
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
    # visualize
    color = [COLOR_MAP[k] for k in KEYPOINT_NAMES]
    vis_image(resized, ax=ax1)
    print(z_com, z_com-crop3dD/2, z_com+crop3dD/2)
    normalized = normalize_depth(resized, z_com, z_size=crop3dD)
    vis_image(normalized, ax=ax2)
    vis_point(point=vu, ax=ax1, color=color)
    vis_point(point=vu, ax=ax2, color=color)
    cropped_zyx = joints_zyx-zyx_com
    vis_point(point=[cropped_zyx], ax=ax3, color=color)
    edge_color = [COLOR_MAP[s, t] for s, t in EDGES]
    vis_edges(point=vu, indices=EDGES, color=edge_color, ax=ax1)
    vis_edges(point=vu, indices=EDGES, color=edge_color, ax=ax2)
    vis_edges(point=[cropped_zyx], indices=EDGES, color=edge_color, ax=ax3)


interact(
    crop_around_3d_center,
    subject_id=SUBJECT_INDICES,
    action=ACTION_LIST,
    seq_idx=SEQ_INDICES,
    frame_id=range(100)
)
