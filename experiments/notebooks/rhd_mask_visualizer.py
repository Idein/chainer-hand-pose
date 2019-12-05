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

# +
import os
import pickle

from chainercv.visualizations import vis_image
from chainercv.utils import read_image
from matplotlib import pyplot as plt
import numpy as np
# -

# load annotations of this set
data_type = "/root/dataset/RHD_published_v2/training"
with open(os.path.join(data_type, 'anno_training.pickle'), 'rb') as f:
    anno_all = pickle.load(f)

sample_id = 0
anno = anno_all[sample_id]
file_format = "{:05d}.png".format(sample_id)
img_file = os.path.join(data_type, "color", file_format)
mask_file = os.path.join(data_type, "mask", file_format)
depth_file = os.path.join(data_type, "depth", file_format)

vis_image(255*read_image(mask_file, dtype=np.uint8))

mask_image = np.squeeze(read_image(mask_file, dtype=np.uint8, color=False))

# # Mask
#
# ```
# Segmentation masks available:
# 0: background, 1: person, 
# 2-4: left thumb [tip to palm], 5-7: left index, ..., 14-16: left pinky, 17: palm, 
# 18-20: right thumb, ..., right palm: 33
# ```

# +
ONESIDE_KEYPOINT_NAMES=[]
for k in ["wrist","thumb", "index", "middle", "ring", "little"]:
    if k=="wrist":
        joint_name="_".join([k])
        ONESIDE_KEYPOINT_NAMES.append(joint_name)
    else:
        for p in ["tip", "dip", "pip", "mcp"]:
            joint_name = "_".join([ k, p])
            ONESIDE_KEYPOINT_NAMES.append(joint_name)

KEYPOINT_NAMES=[]
for hand_side in ["left", "right"]:
    for k in ["wrist","thumb", "index", "middle", "ring", "little"]:
        if k=="wrist":
            joint_name="_".join([hand_side,k])
            KEYPOINT_NAMES.append(joint_name)
        else:
            for p in ["tip", "dip", "pip", "mcp"]:
                joint_name = "_".join([hand_side, k, p])
                KEYPOINT_NAMES.append(joint_name)
                
# (R,G,B)
BASE_COLOR = {
    "thumb": (255, 0, 0),
    "index": (0, 255, 0),
    "middle": (0, 0, 255),
    "ring": (255, 0, 255),
    "little": (255, 255, 0),
    "wrist": (50, 50, 50),
}
# convert value to numpy.array
BASE_COLOR = {k: np.array(v) for k, v in BASE_COLOR.items()}

MASK_COLOR = {
    "background": (0, 0, 0),
    "person": (0, 28, 160),
    "left_palm": (145, 255, 123),
    "right_palm": (128, 14, 0),
}

MASK_NAMES = ["background", "person"]
for hand_side in ["left", "right"]:
    for k in ["thumb", "index", "middle", "ring", "little"]:
        for p in ["tip_dip", "dip_pip", "pip_mcp"]:
            mask_name = "_".join([hand_side, k, p])
            MASK_NAMES.append(mask_name)
            MASK_COLOR[mask_name] = BASE_COLOR[k]
    else:
        mask_name = "_".join([hand_side, "palm"])
        MASK_NAMES.append(mask_name)
# -

imH, imW = mask_image.shape
canvas = np.zeros((imH, imW, 3), dtype=np.uint8)
print(mask_image.shape)
for i, mask_name in enumerate(MASK_NAMES):
    canvas[np.where(mask_image == i)] = MASK_COLOR[mask_name]

plt.imshow(canvas)

kinematic_chain_list = [0,
                        4, 3, 2, 1,
                        8, 7, 6, 5,
                        12, 11, 10, 9,
                        16, 15, 14, 13,
                        20, 19, 18, 17]
[ONESIDE_KEYPOINT_NAMES[k] for k in kinematic_chain_list]
