from math import sin, cos

import chainercv
from chainer.backends.cuda import get_array_module
import numpy as np

"""
Coordinates are ordered as (z, y, x) in 3 dimensional space
It is compatible with ChainerCV project
See:
https://github.com/chainer/chainercv#data-conventions
"""
DATA_CONVENTION = "ZYX"


def calc_com(pts, z=None, axis=0):
    """
    calculate center of mass for given points pts
    """
    xp = get_array_module(pts)
    if z is None:
        return xp.mean(pts, axis=axis)
    return xp.mean(pts, axis=axis), xp.mean(z, axis=axis)


def normalize_joint_zyx(joint, camera, z_size):
    vu, z = camera.zyx2vu(joint, return_z=True)
    vu_com, z_com = calc_com(vu, z)
    zyx_com = camera.vu2zyx(
        vu_com[np.newaxis],
        z_com[np.newaxis],
    ).squeeze()
    z_half = z_size / 2
    return (joint - zyx_com) / z_half


def normalize_joint(joint, edge, root=0, return_scale=False):
    joint = joint - joint[root]
    s, t = edge
    sj, tj = joint[s], joint[t]
    diff = sj - tj
    scale = np.sqrt(diff.dot(diff)) + 1e-8
    joint = joint / scale
    if return_scale:
        return joint, scale
    return joint


def relu(x):
    return max(0, x)


def intersection(bbox0, bbox1):
    y0m, x0m, y0M, x0M = bbox0
    y1m, x1m, y1M, x1M = bbox1
    w = relu(min(x0M, x1M) - max(x0m, x1m))
    h = relu(min(y0M, y1M) - max(y0m, y1m))
    return h, w


def crop_domain2d(image, domain, fill=0):
    C, H, W = image.shape
    bbox0 = domain
    y0m, x0m, y0M, x0M = bbox0
    bbox1 = [0, 0, H, W]
    y1m, x1m, y1M, x1M = bbox1
    outH = y0M - y0m
    outW = x0M - x0m
    h, w = intersection(bbox0, bbox1)
    if h * w == 0:
        new_canvas = np.empty((C, outH, outW), dtype=image.dtype)
        new_canvas[:] = np.array(fill).reshape((-1, 1, 1))
        param = {
            "y_offset": -y0m,
            "x_offset": -x0m,
        }
        return new_canvas, param
    else:
        new_canvas = np.empty((C, outH, outW), dtype=image.dtype)
        new_canvas[:] = np.array(fill).reshape((-1, 1, 1))
        sx, sy = max(x0m, x1m), max(y0m, y1m)
        y_slice, x_slice = slice(sy, sy + h), slice(sx, sx + w)
        cropped = image[:, y_slice, x_slice]
        y_offset = relu(y1m - y0m)
        x_offset = relu(x1m - x0m)
        y_canvas_slice = slice(y_offset, y_offset + h)
        x_canvas_slice = slice(x_offset, x_offset + w)
        new_canvas[:, y_canvas_slice, x_canvas_slice] = cropped
        y_offset = -sy + y_offset
        x_offset = -sx + x_offset
        param = {
            "y_offset": y_offset,
            "x_offset": x_offset,
        }
        return new_canvas, param


def crop_domain3d(image, joint_zyx, camera, domain3d, aug_param):
    vu, z = camera.zyx2vu(joint_zyx, return_z=True)
    scale = 1.0
    if aug_param.get("do_oscillate", False):
        scale = np.random.choice(aug_param["scale_range"])
    print("scale", scale)
    _, crop3dH, crop3dW = scale * np.array(domain3d)

    vu_com, z_com = calc_com(vu, z)

    zyx_com = camera.vu2zyx(
        vu_com[np.newaxis],
        z_com[np.newaxis],
    ).squeeze()

    z_com, y_com, x_com = zyx_com
    xmin = x_com - crop3dW / 2
    ymin = y_com - crop3dH / 2
    xmax = xmin + crop3dW
    ymax = ymin + crop3dH
    [
        [vmin, umin],
        [vmax, umax],
    ] = camera.zyx2vu(np.array([
        [z_com, ymin, xmin],
        [z_com, ymax, xmax],
    ])).astype(int)

    # random oscillation
    if aug_param.get("do_oscillate", False):
        uscale = np.random.choice(aug_param["shift_range"])
        vscale = np.random.choice(aug_param["shift_range"])
        vshift = int(vscale * (vmax - vmin))
        ushift = int(uscale * (umax - umin))
    else:
        vshift = ushift = 0

    domain = [
        vmin + vshift,
        umin + ushift,
        vmax + vshift,
        umax + ushift,
    ]

    cropped, crop_param = crop_domain2d(image, domain)

    camera = camera.translate_camera(
        y_offset=crop_param["y_offset"],
        x_offset=crop_param["x_offset"]
    )

    return cropped, camera


def flip_point_xyz(camera, xyz, size, y_flip=False, x_flip=False):
    uv, z = camera.xyz2uv(xyz, return_z=True)
    vu = uv[:, ::-1]
    W, H = size
    flipped_vu = chainercv.transforms.flip_point(
        vu[np.newaxis],
        (H, W),
        y_flip=y_flip,
        x_flip=x_flip,
    )
    flipped_uv = np.squeeze(flipped_vu)[:, ::-1]
    flipped_xyz = camera.uv2xyz(flipped_uv, z)
    return flipped_xyz


def flip_point_zyx(camera, zyx, size, y_flip=False, x_flip=False):
    vu, z = camera.zyx2vu(zyx, return_z=True)
    H, W = size
    flipped_vu = chainercv.transforms.flip_point(
        vu[np.newaxis],
        (H, W),
        y_flip=y_flip,
        x_flip=x_flip,
    )
    flipped_vu = np.squeeze(flipped_vu)
    flipped_zyx = camera.vu2zyx(flipped_vu, z)
    return flipped_zyx


def rotate_point_uv(uv, angle, center_uv):
    ndim = uv.ndim
    if ndim == 3:
        uv = uv.squeeze()
    c_u, c_v = center_uv
    theta = np.deg2rad(angle)
    rmat_uv = np.array([
        [cos(theta), -sin(theta)],
        [sin(theta), cos(theta)],
    ], dtype=uv.dtype).transpose()
    uv = uv - center_uv
    rot_uv = uv @ rmat_uv
    rot_uv = rot_uv + center_uv
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
        [sin(theta), cos(theta)],
    ], dtype=vu.dtype).transpose()
    rmat_vu = P @ rmat @ P
    vu = vu - center_vu
    rot_vu = vu @ rmat_vu
    rot_vu = rot_vu + center_vu
    if ndim == 3:
        rot_vu = np.expand_dims(rot_vu, axis=0)
    return rot_vu


def rotate_point_zyx(camera, zyx, angle, center_vu):
    vu, z = camera.zyx2vu(zyx, return_z=True)
    rot_vu = rotate_point_vu(vu, angle, center_vu)
    zyx = camera.vu2zyx(rot_vu, z)
    return zyx


def rodrigues(rotation_vector):
    theta = np.linalg.norm(rotation_vector)
    import math
    rv = rotation_vector / theta
    rr = np.array([[rv[i] * rv[j] for j in range(3)] for i in range(3)])
    R = math.cos(theta) * np.eye(3)
    R += (1 - math.cos(theta)) * rr
    R += math.sin(theta) * np.array([
        [0, -rv[2], rv[1]],
        [rv[2], 0, -rv[0]],
        [-rv[1], rv[0], 0],
    ])
    return R
