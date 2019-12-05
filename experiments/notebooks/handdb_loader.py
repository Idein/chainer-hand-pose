# ---
# jupyter:
#   jupytext:
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
import os
import glob
import json
import numpy as np
import chainercv
from chainercv.visualizations import vis_image
from matplotlib import pyplot as plt

from ipywidgets import interact
# -

dataset_dir = os.path.expanduser("~/dataset/handdb_dataset")

# # Manual annotation

# +
manual_annotations_dir = os.path.join(dataset_dir, "hand_labels")
manual_train_dir = os.path.join(manual_annotations_dir, "manual_train")
json_files = sorted(glob.glob(os.path.join(manual_train_dir, "*.json")))


def visualize_result(idx):
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    file = json_files[idx]
    with open(file, 'r') as f:
        anno = json.load(f)
        print(anno)
        joint=np.array(anno["hand_pts"])
        is_left=anno["is_left"]
    img_path = os.path.splitext(file)[0]+".jpg"
    img = chainercv.utils.read_image(img_path)
    vis_image(img, ax=ax)
    umin = joint[:, 0].min()
    umax = joint[:, 0].max()
    ulen = umax-umin
    vmin = joint[:, 1].min()
    vmax = joint[:, 1].max()
    vlen = vmax-vmin
    boxlen = int(1.5*max(ulen,vlen))
    uc, vc = (umax+umin)/2, (vmax+vmin)/2
    uc = int(uc)
    vc = int(vc)
    _,imH,imW=img.shape
    vmin=max(0,vc-boxlen//2)
    vmax=min(imH,vc+boxlen//2)
    umin=max(0,uc-boxlen//2)
    umax=min(imW,uc+boxlen//2)
    vis_image(img[:,vmin:vmax,umin:umax], ax=ax2)
    joint= joint[:,:2]-np.array([[umin,vmin]])
    ax2.scatter(*joint.transpose())


interact(visualize_result,idx=range(100))
# -

# # Synth hand

manual_annotations_dir = os.path.join(dataset_dir, "hand_labels_synth")
# exclude synth1 cuz these files under synth1 has no 21 points
manual_train_dir = os.path.join(manual_annotations_dir, "synth4")
json_files = sorted(glob.glob(os.path.join(manual_train_dir, "*.json")))

# +
manual_annotations_dir = os.path.join(dataset_dir, "hand_labels_synth")
# exclude synth1 cuz these files under synth1 has no 21 points

def visualize_synth(synth, idx):
    fig=plt.figure(figsize=(10,5))
    ax1=fig.add_subplot(121)
    ax2=fig.add_subplot(122)
    manual_train_dir = os.path.join(manual_annotations_dir, "synth{}".format(synth))
    json_files = sorted(glob.glob(os.path.join(manual_train_dir, "*.json")))
    file=json_files[idx]
    with open(file,'r') as f:
        anns=json.load(f)
        print(anns)
    joint=np.array(anns["hand_pts"])
    print(joint.shape)
    is_left=anns["is_left"]
    print(file)
    imfile = file.replace("json","jpg")
    print(imfile)
    import imageio
    img=chainercv.utils.read_image(imfile)
    vis_image(img,ax=ax1)
    umin = joint[:, 0].min()
    umax = joint[:, 0].max()
    ulen = umax-umin
    vmin = joint[:, 1].min()
    vmax = joint[:, 1].max()
    vlen = vmax-vmin
    boxlen = int(1.5*max(ulen,vlen))
    uc, vc = (umax+umin)/2, (vmax+vmin)/2
    uc = int(uc)
    vc = int(vc)
    _,imH,imW=img.shape
    vmin=max(0,vc-boxlen//2)
    vmax=min(imH,vc+boxlen//2)
    umin=max(0,uc-boxlen//2)
    umax=min(imW,uc+boxlen//2)
    vis_image(img[:,vmin:vmax,umin:umax], ax=ax2)
    joint= joint[:,:2]-np.array([[umin,vmin]])
    plt.scatter(*joint.transpose())
        
interact(visualize_synth,synth=[2,3,4], idx=82)
# -

# # 143

panoptic = os.path.join(dataset_dir, "hand143_panopticdb")
# exclude synth1 cuz these files under synth1 has no 21 points
json_file = os.path.join(panoptic, "hands_v143_14817.json")
with open(json_file, 'r') as f:
    annotations = json.load(f)

annotations["root"][0].keys()
annotations["root"][0]

# +
panoptic = os.path.join(dataset_dir, "hand143_panopticdb")
# exclude synth1 cuz these files under synth1 has no 21 points
json_file = os.path.join(panoptic, "hands_v143_14817.json")
with open(json_file, 'r') as f:
    annotations = json.load(f)

def visualize_result(idx):
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    anno = annotations["root"][idx]
    print(anno)
    img_path = os.path.join(panoptic,anno["img_paths"])
    img=chainercv.utils.read_image(img_path)
    vis_image(img,ax=ax)
    joint=np.array(anno["joint_self"])
    joint=joint[:,:2]
    ax.scatter(*joint.transpose())
    umin = joint[:, 0].min()
    umax = joint[:, 0].max()
    ulen = umax-umin
    vmin = joint[:, 1].min()
    vmax = joint[:, 1].max()
    vlen = vmax-vmin
    boxlen = int(1.5*max(ulen,vlen))
    uc, vc = (umax+umin)/2, (vmax+vmin)/2
    uc = int(uc)
    vc = int(vc)
    _,imH,imW=img.shape
    vmin=max(0,vc-boxlen//2)
    vmax=min(imH,vc+boxlen//2)
    umin=max(0,uc-boxlen//2)
    umax=min(imW,uc+boxlen//2)
    vis_image(img[:,vmin:vmax,umin:umax], ax=ax2)
    joint= joint[:,:2]-np.array([[umin,vmin]])
    ax2.scatter(*joint.transpose())
interact(visualize_result,idx=range(100,300))
# -

yx = 100*np.random.random((21,2))
zyx=np.ones((21,3))
zyx[:,1:]=yx

# +
import sys
sys.path.append("../../src")
import camera

c=camera.CameraIntr(fx=1,fy=1,u0=0,v0=0)
vu=c.zyx2vu(zyx)
# -

plt.scatter(*yx.transpose())

plt.scatter(*vu.transpose())


