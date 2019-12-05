# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import json
import os

import chainercv
import imageio
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from ipywidgets import interact

# +
import itertools
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

DEFAULT_KEYPOINT_NAMES=[]

for k in ["root", "thumb", "index", "middle", "ring", "little"]:
    if k == "root":
        joint_name = "_".join([k])
        DEFAULT_KEYPOINT_NAMES.append(joint_name)
    else:
        for p in ["tip", "dip", "pip", "mcp"]:
            joint_name = "_".join([k, p])
            DEFAULT_KEYPOINT_NAMES.append(joint_name)
            
EDGE_NAMES = []
for f in ["thumb", "index", "middle", "ring", "little"]:
    for s, t in pairwise(["root", "tip", "dip", "pip", "mcp"]):
        if s == "root":
            t = "_".join([f, t])
        else:
            s = "_".join([f, s])
            t = "_".join([f, t])
        EDGE_NAMES.append([s, t])

EDGES = [[DEFAULT_KEYPOINT_NAMES.index(s), DEFAULT_KEYPOINT_NAMES.index(t)]
         for s, t in EDGE_NAMES]

colors=np.random.choice(range(0,255),15).astype(int)
COLOR_MAP={}
MASK_VALUE=[1, 2, 3, 4, 5, 6, 7, 248, 249, 250, 251, 252, 253, 254, 255]
for i,v in enumerate(MASK_VALUE):
    COLOR_MAP[v]=colors[i]
else:
    COLOR_MAP[255]=255
# -

dataset_dir = os.path.expanduser("~/dataset/FreiHAND_pub_v1")
with open(os.path.join(dataset_dir,"training_xyz.json"),'r') as f:
    training_xyz=json.load(f)
with open(os.path.join(dataset_dir,"training_K.json"),'r') as f:
    training_K=json.load(f)

# +
# %matplotlib notebook


def visualize_dataset(idx):
    fig=plt.figure(figsize=(8,8))
    ax=fig.add_subplot(221)
    ax2=fig.add_subplot(222)
    ax3=fig.add_subplot(223,projection="3d")
    joint_xyz=np.array(training_xyz[idx])
    K = np.array(training_K[idx])
    uv=joint_xyz @ K.transpose()
    uv = uv/uv[:,2:]
    uv = uv[:,:2]
    img_path=os.path.join(dataset_dir,"training","rgb","{:08d}.jpg".format(idx))
    mask_path=os.path.join(dataset_dir,"training","mask","{:08d}.jpg".format(idx))
    img=chainercv.utils.read_image(img_path)
    mask=chainercv.utils.read_image(mask_path,dtype=np.uint8)
    C,H,W=mask.shape
    color_mask=np.zeros((C,H,W)).astype(np.uint8)
    for i in MASK_VALUE:
        color_mask[mask==i]=COLOR_MAP[i]
    ax.scatter(*uv.transpose())
    chainercv.visualizations.vis_image(img,ax=ax)
    chainercv.visualizations.vis_image(color_mask,ax=ax2)

    ax3.scatter(*joint_xyz.transpose())
    for s,t in EDGES:
        edge = uv[[s,t]]
        ax.plot(*edge.transpose())
        edge=joint_xyz[[s,t]]
        ax3.plot(*edge.transpose())
        
    for ax in [ax3]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(-65, -90)

levels=len(MASK_VALUE)
samples=np.random.choice(len(training_xyz),500)
interact(visualize_dataset,idx=samples)
