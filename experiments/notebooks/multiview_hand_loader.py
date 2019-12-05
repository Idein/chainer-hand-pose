# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# +
import os
import glob

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import chainercv
from chainercv.visualizations import vis_image

from ipywidgets import interact
# -

dataset_dir = os.path.expanduser("~/dataset/multiview_hand")

annotated_dirs=sorted(os.listdir(os.path.join(dataset_dir,"annotated_frames")))
augumented_dirs=sorted(os.listdir(os.path.join(dataset_dir,"augmented_samples")))
calibrations_dirs=sorted(os.listdir(os.path.join(dataset_dir,"calibrations")))
sample_dir = annotated_dirs[0]
texts=sorted(glob.glob(os.path.join(os.path.join(dataset_dir,"annotated_frames"),sample_dir,"*_joints.txt")))

# +
KEYPOINT_NAMES = [
    'F4_KNU1_A',
    'F4_KNU1_B',
    'F4_KNU2_A',
    'F4_KNU3_A',
    'F3_KNU1_A',
    'F3_KNU1_B',
    'F3_KNU2_A',
    'F3_KNU3_A',
    'F1_KNU1_A',
    'F1_KNU1_B',
    'F1_KNU2_A',
    'F1_KNU3_A',
    'F2_KNU1_A',
    'F2_KNU1_B',
    'F2_KNU2_A',
    'F2_KNU3_A',
    'TH_KNU1_A',
    'TH_KNU1_B',
    'TH_KNU2_A',
    'TH_KNU3_A',
    'PALM_POSITION',
    'PALM_NORMAL'
]

EDGE_NAMES=[]

import itertools

def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

for f in ["TH","F1","F2","F3","F4"]:
    for p,q in pairwise(["PALM_POSITION","KNU1_B","KNU1_A","KNU2_A","KNU3_A"]):
        if p != "PALM_POSITION":
            EDGE_NAMES.append(["_".join([f,p]),"_".join([f,q])])
        else:
            EDGE_NAMES.append([p,"_".join([f,q])])
EDGES = [(KEYPOINT_NAMES.index(s),KEYPOINT_NAMES.index(t)) for (s,t) in EDGE_NAMES]
# -

import pandas as pd
sample_texts=0
df=pd.read_csv(texts[sample_texts], sep=' ', header=None)

import pandas as pd
df=pd.read_csv(texts[sample_texts], sep=' ',usecols=[1,2,3], header=None)
joint=df.to_numpy()

joint.shape

# +
# %matplotlib notebook
fig=plt.figure()
ax=fig.add_subplot(111,projection="3d")
for xyz in joint:
    ax.scatter(*xyz)
for s,t in EDGES:
    xyz=joint[[s,t]]
    ax.plot(*xyz.transpose())

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.view_init(65,90)

# +
import pickle
import cv2


def getCameraMatrix():
    Fx = 614.878
    Fy = 615.479
    Cx = 313.219
    Cy = 231.288
    cameraMatrix = np.array([[Fx, 0, Cx],
                             [0, Fy, Cy],
                             [0, 0, 1]])
    return cameraMatrix

cam_intr = getCameraMatrix()
distCoeffs=np.array([0.092701, -0.175877, -0.0035687, -0.00302299, 0])

def visualize(data_idx, frame_idx, cam_idx):
    data_folder = annotated_dirs[data_idx]
    img_path = os.path.join(dataset_dir,"annotated_frames", data_folder,
                            "{}_webcam_{}.jpg".format(frame_idx, cam_idx))
    img_path = os.path.join(dataset_dir,"augmented_samples", data_folder,
                            "{}_webcam_{}.jpg".format(frame_idx, cam_idx))
    joint_path = os.path.join(dataset_dir,"annotated_frames", data_folder,
                              "{}_joints.txt".format(frame_idx))
    df = pd.read_csv(joint_path, sep=' ', usecols=[1, 2, 3], header=None)
    joint = df.to_numpy()
    joint = joint[:21] # ignore last keypoint i.e. PALM_NORMAL
    calib_dir = os.path.join(dataset_dir,"calibrations", data_folder,
                             "webcam_{}".format(cam_idx))
    print(img_path,joint_path,calib_dir)
    with open(os.path.join(calib_dir, "rvec.pkl"), 'rb') as f:
        calibR = pickle.load(f, encoding='latin1')
    with open(os.path.join(calib_dir, "tvec.pkl"), 'rb') as f:
        calibT = pickle.load(f, encoding='latin1')
    R, _ = cv2.Rodrigues(calibR)
    joint = joint.dot(R.transpose()) + calibT.transpose()
    print(joint.shape)
    #uv, _ = cv2.projectPoints(joint, calibR, calibT, cam_intr, distCoeffs)
    uv = joint.dot(cam_intr.transpose())
    uv = uv/uv[:,2:]
    uv = uv[:,:2]
    print(uv.shape)
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax3 = fig.add_subplot(122, projection="3d")

    img = chainercv.utils.read_image(img_path)
    vis_image(img, ax=ax)
    for xyz in joint:
        ax3.scatter(*xyz)
    ax.scatter(*uv.transpose())
    for s, t in EDGES:
        xyz = joint[[s, t]]
        ax3.plot(*xyz.transpose())
        
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.view_init(-65, -90)


interact(visualize, data_idx=range(22),
         frame_idx=range(100), cam_idx=[1, 2, 3, 4])
# -


