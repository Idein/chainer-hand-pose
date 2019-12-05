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

# # ChainerCV3D transformer
# - 3D の座標点の変換ロジックを作成する
# - 既存の `chainercv` のインターフェースにcompatibleになるような設計を目指す
#     - これは３次元座標を2次元の投影面に投影した際の座標系が `ChainerCV` の $(y,x)$ と整合性が持つと良いと考えるため.

# # import modules

import numpy as np
dtype = np.float32

# # camera
# - intrinsic
# $$
# \begin{bmatrix}
# f_x & 0 & u_0 \\
# 0 & f_y & v_0 \\
# 0 & 0 & 1 \\
# \end{bmatrix}
# $$
# - extrinsic

# # define camera intrinsic matrix

# ## Permutate xyz <-> zyx
#
# $$
# \begin{bmatrix}
# 0 & 0 & 1 \\
# 0 & 1 & 0 \\
# 1 & 0 & 0 \\
# \end{bmatrix}
# \begin{bmatrix}
# x\\y\\z
# \end{bmatrix}
# =
# \begin{bmatrix}
# z\\y\\x
# \end{bmatrix}
# $$

# ## define perspective model

# +
"""
                       Z
                       *
                      /
    o|      u_0      /
    ----------------------* u
     |       |     /
     |       |    /
     |       |   /
     |       |  /
  v_0|-------$ /
     |        /  
     |       /
     *      /
     v     /O 
         ---|--------------* X
            |
            |
            |
            |
            *
            Y
"""

"""
                       *X
                       |x
                       |
                       |
              $--------|u
              |        |
              |        |
              |        |
----0--------f_x-------z--------*Z
u=f_x*x/z
"""

"""
                       *Y
                       |y
                       |
                       |
              $--------|v
              |        |
              |        |
              |        |
----0--------f_x-------z--------*Z
v=f_y*y/z
"""

sk = 0  # skew
# Image center
u0 = 935.732544
v0 = 540.681030
# Focal Length
fx = 1395.749023
fy = 1395.749268

# camera matrix for standard order (x,y,z)
cam_intr_xyz = np.array([
    [fx, sk, u0],
    [0,  fy, v0],
    [0,   0,  1],
], dtype=dtype).transpose()

Pzx = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
], dtype=dtype).transpose()

Pxz = Pzx

# camera matrix for generalized chainercv order (z,y,x)
cam_intr_zyx = Pxz@cam_intr_xyz@Pzx
# -

np.random.randint

cam_extr = np.array([
    [0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
    [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
    [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
    [0, 0, 0, 1],
], dtype=dtype).transpose()


def world_xyz2xyz(world_xyz):
    nk, *_ = world_xyz.shape
    hom_world_xyz = np.concatenate([skel, np.ones(nk, 1)], axis=1)
    hom_cam_xyz = hom_world_xyz @ cam_extr
    cam_xyz = hom_cam_xyz[:, :3]
    return cam_xyz


# +
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
        return vu, z
    return vu


# +
xyz = np.array([
    [1, 3, 4],
    [9, 2, 5],
    [9, 2, 5],
    [9, 2, 5]
])

zyx = xyz[:, ::-1]
ret_xyz, z = xyz2uv(xyz, return_z=True)
ret_zyx, z = zyx2vu(zyx, return_z=True)
np.allclose(ret_xyz, ret_zyx[:, ::-1])
# -

# ## restore 3D cordinate such that its perspective is uv coordinate

# +
pull_back_xyz = np.array([
    [1/fx, 0, -u0/fx],
    [0, 1/fy, -v0/fy],
    [0, 0, 1],
], dtype=dtype).transpose()

pull_back_zyx = Pxz @ pull_back_xyz @ Pzx


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


# +
xyz = np.array([
    [1, 3, 4],
    [9, 2, 5],
    [3, 2, 1],
    [3, 4, 5]
], dtype=dtype)

expected = np.copy(xyz)
z = xyz[:, 2:]
uv = xyz2uv(xyz)
ret = uv2xyz(uv, z)

assert np.allclose(ret, expected)

# +
zyx = xyz[::-1]
expected = np.copy(zyx)
z = zyx[:, :1]
vu = zyx2vu(zyx)
ret = vu2zyx(vu, z)

assert np.allclose(ret, expected)
# -

# # rotate points on perspective plane and calc pull back

# +
from math import sin, cos


def rot_xyz_around_uv(xyz, angle, center_uv):
    theta = np.deg2rad(angle)
    uv, z = xyz2uv(xyz, return_z=True)
    c_u, c_v = center_uv
    rot_around_uv = np.array([
        [cos(theta), -sin(theta), c_u],
        [sin(theta),  cos(theta), c_v],
        [0, 0, 1],
    ], dtype=dtype).transpose()
    uv = uv-center_uv
    nk, *_ = uv.shape
    hom_uv = np.concatenate([uv, np.ones((nk, 1), dtype=dtype)], axis=1)
    rot_uvz = hom_uv @ rot_around_uv
    rot_uv = rot_uvz[:, :2]
    xyz = uv2xyz(rot_uv, z)
    return xyz


def rot_zyx_around_vu(zyx, angle, center_vu):
    theta = np.deg2rad(angle)
    vu, z = zyx2vu(zyx, return_z=True)
    c_v, c_u = center_vu
    rot_around_vu = np.array([
        [cos(theta), -sin(theta), c_u],
        [sin(theta),  cos(theta), c_v],
        [0, 0, 1],
    ], dtype=dtype).transpose()
    vu = vu-center_vu
    nk, *_ = vu.shape
    hom_vu = np.concatenate([np.ones((nk, 1), dtype=dtype), vu], axis=1)
    rot_zvu = hom_vu @ Pzx @ rot_around_vu @ Pxz
    rot_vu = rot_zvu[:, 1:]
    zyx = vu2zyx(rot_vu, z)
    return zyx


# +
import random

angle = np.random.randint(360)
xyz = np.array(np.random.random((5, 3)))
center_uv = np.random.random(2)
print("angle", angle)
print("xyz", xyz, "\n")
rot = rot_xyz_around_uv(xyz, angle, center_uv=center_uv)
# inverse
inv = rot_xyz_around_uv(rot, -angle, center_uv=center_uv)

assert np.allclose(xyz, inv), (xyz, inv)

# +
angle = np.random.randint(360)
zyx = np.array(np.random.random((5, 3)))
center_vu = np.random.random(2)
print("angle", angle)
print("zyx", zyx, "\n")
rot = rot_zyx_around_vu(zyx, angle, center_vu=center_vu)
# inverse
inv = rot_zyx_around_vu(rot, -angle, center_vu=center_vu)


assert np.allclose(zyx, inv), (zyx, inv)
# -

# # Flip around vertical axis

# +
from chainercv import transforms


def flip_point_xyz(xyz, size, x_flip=False, y_flip=False):
    uv, z = xyz2uv(xyz, return_z=True)
    vu = uv[:, ::-1]
    W, H = size
    flipped_vu = transforms.flip_point(
        vu[np.newaxis],
        (H, W),
        x_flip=x_flip,
        y_flip=y_flip
    )
    flipped_uv = np.squeeze(flipped_vu)[:, ::-1]
    flipped_xyz = uv2xyz(flipped_uv, z)
    return flipped_xyz


def flip_point_zyx(zyx, size, x_flip=False, y_flip=False):
    vu, z = zyx2vu(zyx, return_z=True)
    H, W = size
    flipped_vu = transforms.flip_point(
        vu[np.newaxis],
        (H, W),
        x_flip=x_flip,
        y_flip=y_flip
    )
    flipped_vu = np.squeeze(flipped_vu)
    flipped_zyx = vu2zyx(np.squeeze(flipped_vu), z)
    return flipped_zyx


# +
xyz = np.array([
    [1, 2, 3],
    [3, 4, 5],
    [4, 5, 6],
    [4, 3, 2],
], dtype=dtype)

print(xyz.shape)
proj_shape = (100, 100)
flipped_xyz = flip_point_xyz(xyz, proj_shape, x_flip=True, y_flip=False)
ret_xyz = flip_point_xyz(flipped_xyz, proj_shape, x_flip=True, y_flip=False)
assert np.allclose(ret_xyz, xyz)

# +
zyx = xyz[:, ::-1]

flipped_zyx = flip_point_zyx(zyx, [100, 100], x_flip=True, y_flip=False)
ret_zyx = flip_point_zyx(flipped_zyx, [100, 100], x_flip=True, y_flip=False)
assert np.allclose(ret_zyx, zyx)
assert np.allclose(ret_zyx, ret_xyz[:, ::-1])

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


# +
class CameraIntr():
    def __init__(self, u0, v0, fx, fy, sk=0, dtype=np.float32):
        cam_intr_xyz = np.array([
            [fx, sk, u0],
            [0,  fy, v0],
            [0,   0,  1],
        ], dtype=dtype).transpose()

        pull_back_xyz = np.array([
            [1/fx, 0, -u0/fx],
            [0, 1/fy, -v0/fy],
            [0,    0,      1],
        ], dtype=dtype).transpose()

        # convert xyz -> zyx and vice versa
        P = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ], dtype=dtype).transpose()

        self.cam_intr_xyz = cam_intr_xyz
        self.pull_back_xyz = pull_back_xyz
        self.cam_intr_zyx = P @ cam_intr_xyz  @ P
        self.pull_back_zyx = P @ pull_back_xyz @ P

    def xyz2uv(self, xyz, return_z=False):
        z = xyz[:, 2:]
        uv_ = xyz/z @ self.cam_intr_xyz
        uv = uv_[:, :2]
        if return_z:
            return uv, z
        return uv

    def zyx2vu(self, zyx, return_z=False):
        z = zyx[:, :1]
        zvu = zyx/z @ self.cam_intr_zyx
        vu = zvu[:, 1:]
        if return_z:
            return vu, z
        return vu

    def uv2xyz(self, uv, z):
        nk, *_ = uv.shape
        hom_uv = np.concatenate([uv, np.ones((nk, 1), dtype=dtype)], axis=1)
        xy_ = hom_uv @ self.pull_back_xyz
        xyz = z*xy_
        return xyz

    def vu2zyx(self, vu, z):
        nk, *_ = vu.shape
        hom_vu = np.concatenate([np.ones((nk, 1), dtype=dtype), vu], axis=1)
        _yx = hom_vu @ self.pull_back_zyx
        zyx = z*_yx
        return zyx
    
CameraIntr(**{"u0":u0,"v0":v0,"fx":fx,"fy":fy})


# +
class CameraExtr(object):
    def __init__(self, r, t, dtype=np.float32):
        _tr_concat = np.concatenate([r, t.reshape(3, 1)], axis=1)
        cam_extr_xyz = np.concatenate(
            [_tr_concat, np.zeros((1, 4))], axis=0).transpose()
        # xyzw->zyxw and vice versa
        P = np.array([
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ], dtype=dtype).transpose()
        cam_extr_zyx = P @ cam_extr_xyz @ P
        self.cam_extr_xyz = cam_extr_xyz
        self.cam_extr_zyx = cam_extr_zyx

    def world_xyz2cam_xyz(self, world_xyz):
        nk, *_ = world_xyz.shape
        hom_world_xyz = np.concatenate([world_xyz, np.ones((nk, 1))], axis=1)
        hom_cam_xyz = hom_world_xyz @ self.cam_extr_xyz
        cam_xyz = hom_cam_xyz[:, :3]
        return cam_xyz

    def world_zyx2cam_zyx():
        nk, *_ = world_zyx.shape
        hom_world_zyx = np.concatenate([np.ones((nk, 1)), world_zyx], axis=1)
        hom_cam_zyx = hom_world_zyx @ self.cam_extr_zyx
        #wzyx -> zyx
        cam_zyx = hom_cam_zyx[:, 1:]
        return cam_zyx

R = np.array([
    [0.999988496304, -0.00468848412856, 0.000982563360594],
    [0.00469115935266, 0.999985218048, -0.00273845880292],
    [-0.000969709653873, 0.00274303671904, 0.99999576807],
])

t = np.array([25.7, 1.22, 3.902])

CameraExtr(R, t)

# +
dtype = np.float32

P = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
], dtype=dtype).transpose()


def translate(vu, offset_vu):
    nk, _ = vu.shape
    hom_vu = np.concatenate([np.ones([nk, 1]), vu], axis=1)
    v0, u0 = offset_vu
    t = np.array([
        [1, 0, u0],
        [0, 1, v0],
        [0, 0, 1],
    ], dtype=dtype).transpose()
    hom_translated = hom_vu @ P @ t @ P
    translated = hom_translated[:, 1:]
    return translated


def scale(vu, scale_vu):
    sv, su = scale_vu
    nk, _ = vu.shape
    hom_vu = np.concatenate([np.ones([nk, 1]), vu], axis=1)
    s = np.array([
        [su, 0, 0],
        [0, sv, 0],
        [0, 0, 1],
    ], dtype=dtype)
    hom_scaled = hom_vu @ P @ s @ P
    scaled = hom_scaled[:, 1:]
    return scaled


# -

x2 = -4
x1 = 3
v0, u0 = 1., 1.
translate(np.array([[x2, x1]]), (v0, u0))

x2 = -4
x1 = 4
v0, u0 = 2, 0.5
scale(np.array([[x2, x1]]), (v0, u0))
