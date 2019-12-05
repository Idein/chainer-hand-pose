import numpy as np
from scipy import stats

VALID_MIN = 20
VALID_MAX = 2 ** (16 - 1)

COLOR_MAP = {
    "hand": (0, 0, 255),
    "left": (255, 0, 0),
    "right": (0, 255, 0),
}


def convert_depth_to_uvd(depth):
    if depth.ndim == 2:
        d = np.expand_dims(depth, axis=0)
    d = depth
    _, H, W = d.shape
    uv = np.meshgrid(range(W), range(H))
    uvd = np.concatenate([uv, d], axis=0)
    return uvd


def define_background(depth):
    valid_loc = np.logical_and(VALID_MIN <= depth, depth <= VALID_MAX)
    # define background as most frequently occurring number i.e. mode
    valid_depth = depth[valid_loc]
    mode_val, mode_num = stats.mode(valid_depth.ravel())
    background = mode_val.squeeze()
    return background


def normalize_depth(depth, z_com, z_size):
    z_half = z_size / 2
    valid_loc = np.logical_and(VALID_MIN <= depth, depth <= VALID_MAX)
    invalid_loc = np.logical_not(valid_loc)
    depth[invalid_loc] = z_com + z_half
    depth[depth > z_com + z_half] = z_com + z_half
    depth[depth < z_com - z_half] = z_com - z_half
    depth -= z_com
    depth = depth / z_half
    return depth


def normalize_rgb(rgb):
    return (rgb / 255.) - 0.5


def denormalize_rgb(rgb):
    return 255. * (rgb + 0.5)
