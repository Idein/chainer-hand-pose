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

# # ICVL dataset

# +
import os
import imageio

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import pandas as pd

from ipywidgets import interact
# -

dataset_dir = os.path.expanduser("~/dataset/icvl_dataset")

df = pd.read_csv(os.path.join(dataset_dir, "labels.txt"), header=None, sep=' ')

# +
import itertools


def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
KEYPOINT_NAMES = ["Palm"]
for f in FINGERS:
    for p in ["root", "mid", "tip"]:
        name = "_".join([f, p])
        KEYPOINT_NAMES.append(name)

EDGE_NAMES = []
for f in FINGERS:
    for s, t in pairwise(["Palm", "root", "mid", "tip"]):
        s_name = "_".join([f, s]) if s != "Palm" else s
        t_name = "_".join([f, t])
        EDGE_NAMES.append([s_name, t_name])

EDGES = [(KEYPOINT_NAMES.index(s), KEYPOINT_NAMES.index(t))
         for s, t in EDGE_NAMES]

columns = ["file_path"]
for k in KEYPOINT_NAMES:
    for c in ["x", "y", "z"]:
        c_k = "_".join([c, k])
        columns.append(c_k)
# -

df.columns = columns

df

# +
# %matplotlib notebook


def visualize_hand(idx):
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    s = df.iloc[idx]
    depth_path = os.path.join(dataset_dir, "Training","Depth", s["file_path"])
    depth = imageio.imread(depth_path)
    print(depth.max(), depth.min())
    im = ax1.imshow(depth)
    #fig.colorbar(im, ax=ax1)
    joints = s[1:].values.reshape(-1, 3).astype(float)
    ax2.scatter(*joints.transpose())
    for s, t in EDGES:
        sx, sy, sz = joints[s]
        tx, ty, tz = joints[t]
        ax2.plot([sx, tx], [sy, ty], [sz, tz])
    fig.tight_layout()
    ax2.view_init(elev=-90., azim=-90)


interact(visualize_hand, idx=range(100))
# -


