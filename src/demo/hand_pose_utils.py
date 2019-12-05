import itertools

import numpy as np

from transforms import flip_point
# Decimal Code (R,G,B)
_BASE_COLOR = {
    "RED": (255, 0, 0),
    "GREEN": (0, 255, 0),
    "BLUE": (0, 0, 255),
    "YELLOW": (255, 255, 0),
    "CYAN": (0, 255, 255),
    "MAGENTA": (255, 0, 255),
}


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


"""
Coordinates are ordered as (z, y, x) in 3 dimensional space
It is compatible with ChainerCV project
See:
https://github.com/chainer/chainercv#data-conventions
"""
DATA_CONVENTION = "ZYX"

NUM_KEYPOINTS = 21
STANDARD_KEYPOINT_NAMES = [
    "root",
    "thumb_mcp",
    "thumb_pip",
    "thumb_dip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "little_mcp",
    "little_pip",
    "little_dip",
    "little_tip",
]

assert len(STANDARD_KEYPOINT_NAMES) == NUM_KEYPOINTS


def make_keypoint_converter(root, fingers, parts, sep="_"):
    converter = {"root": root}
    for f, std_f in zip(fingers, ["thumb", "index", "middle", "ring", "little"]):
        for p, std_p in zip(parts, ["mcp", "pip", "dip", "tip"]):
            converter["_".join([std_f, std_p])] = sep.join([f, p])
    return converter


BASE_COLOR = {
    "root": (50, 50, 50),
    "thumb": _BASE_COLOR["MAGENTA"],
    "index": _BASE_COLOR["BLUE"],
    "middle": _BASE_COLOR["GREEN"],
    "ring": _BASE_COLOR["YELLOW"],
    "little": _BASE_COLOR["RED"],
}

COLOR_MAP = {"root": BASE_COLOR["root"]}

EDGE_NAMES = []
for f in ["thumb", "index", "middle", "ring", "little"]:
    for s, t in pairwise(["root", "mcp", "pip", "dip", "tip"]):
        color = BASE_COLOR[f]
        if s == "root":
            t = "_".join([f, t])
        else:
            s = "_".join([f, s])
            t = "_".join([f, t])
        EDGE_NAMES.append([s, t])
        COLOR_MAP[s, t] = color
        COLOR_MAP[t] = color

EDGES = [[STANDARD_KEYPOINT_NAMES.index(s), STANDARD_KEYPOINT_NAMES.index(t)]
         for s, t in EDGE_NAMES]

for s, t in EDGE_NAMES:
    COLOR_MAP[
        STANDARD_KEYPOINT_NAMES.index(s),
        STANDARD_KEYPOINT_NAMES.index(t)
    ] = COLOR_MAP[s, t]
    COLOR_MAP[STANDARD_KEYPOINT_NAMES.index(s)] = COLOR_MAP[s]
    COLOR_MAP[STANDARD_KEYPOINT_NAMES.index(t)] = COLOR_MAP[t]
# convert value as np.array
COLOR_MAP = {k: np.array(v) for k, v in COLOR_MAP.items()}


def normalize_rgb(rgb):
    return (rgb / 255.) - 0.5


def denormalize_rgb(rgb):
    return 255. * (rgb + 0.5)


def format_kp_proj(point, outH, outW, offsetH=0, offsetW=0, x_flip=False, y_flip=False):
    vmin = np.min(point[:, 0])
    umin = np.min(point[:, 1])
    vmax = np.max(point[:, 0])
    umax = np.max(point[:, 1])
    ulen = vmax - vmin
    vlen = umax - umin
    scale = min(outH, outW) / max(ulen, vlen)
    offset = np.array([vmin, umin])
    point = scale * (point - offset)
    point = flip_point(
        point[np.newaxis],
        (outH, outW),
        x_flip=x_flip,
        y_flip=y_flip,
    ).squeeze(axis=0)
    point = point + np.array([offsetH, offsetW])
    return point
