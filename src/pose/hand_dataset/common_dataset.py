import numpy as np

from pose.utils import pairwise
from pose.visualizations import BASE_COLOR as _BASE_COLOR

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

# define root joint and reference edge. They are used to normalize 3D joint
ROOT_IDX = STANDARD_KEYPOINT_NAMES.index("root")
MMCP = STANDARD_KEYPOINT_NAMES.index("middle_mcp")
MPIP = STANDARD_KEYPOINT_NAMES.index("middle_pip")
LMCP = STANDARD_KEYPOINT_NAMES.index("little_mcp")
REF_EDGE = (MMCP, MPIP)


def wrist2palm(joint):
    wrist_idx = STANDARD_KEYPOINT_NAMES.index("root")
    middle_mcp_idx = STANDARD_KEYPOINT_NAMES.index("middle_mcp")
    parm = (joint[wrist_idx] + joint[middle_mcp_idx]) / 2
    joint[wrist_idx] = parm
    return joint
