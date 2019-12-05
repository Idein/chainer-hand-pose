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

# # Dataset Visualizer

# +
import sys

# Yeah darkside
sys.path.append("../../src")
sys.path.append("../../src/pose/hand_dataset")

# +
import copy
import logging

logger = logging.getLogger(__name__)

from chainer.dataset import DatasetMixin
import chainercv
from chainercv.links.model.ssd import random_distort
import numpy as np

from pose.hand_dataset.geometry_utils import crop_domain2d, crop_domain3d, calc_com
from pose.hand_dataset.geometry_utils import flip_point_zyx, rotate_point_zyx, rotate_point_vu


def load_dataset(dataset_type, visualize=True, iterate_all=False):
    import os
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    enable_rgb = True
    enable_depth = True
    debug = True
    if dataset_type == "fhad":
        dataset_dir = os.path.expanduser("~/dataset/fhad")
        import fhad_dataset as x_dataset
        from fhad_dataset import get_fhad_dataset as get_dataset
    elif dataset_type == "stb":
        dataset_dir = os.path.expanduser("~/dataset/stb")
        import stb_dataset as x_dataset
        from stb_dataset import get_stb_dataset as get_dataset
    elif dataset_type == "rhd":
        dataset_dir = os.path.expanduser("~/dataset/RHD_published_v2")
        import rhd_dataset as x_dataset
        from rhd_dataset import get_rhd_dataset as get_dataset
        debug = False
    elif dataset_type == "msra15":
        dataset_dir = os.path.expanduser("~/dataset/cvpr15_MSRAHandGestureDB")
        import msra15_dataset as x_dataset
        from msra15_dataset import get_msra15_dataset as get_dataset
        enable_rgb = False
    elif dataset_type == "nyu":
        dataset_dir = os.path.expanduser("~/dataset/nyu_hand_dataset_v2")
        import nyu_dataset as x_dataset
        from nyu_dataset import get_nyu_dataset as get_dataset
        enable_rgb = False
    elif dataset_type == "synth":
        dataset_dir = os.path.expanduser("~/dataset/SynthHands_Release")
        import synth_dataset as x_dataset
        from synth_dataset import get_synth_dataset as get_dataset
    elif dataset_type == "ganerated":
        dataset_dir = os.path.expanduser("~/dataset/GANeratedHands_Release")
        import ganerated_dataset as x_dataset
        from ganerated_dataset import get_ganerated_dataset as get_dataset
        enable_depth = False
        debug = True
    elif dataset_type == "multiview":
        dataset_dir = os.path.expanduser("~/dataset/multiview_hand")
        import multiview_dataset as x_dataset
        from multiview_dataset import get_multiview_dataset as get_dataset
        enable_depth = False
        debug = True
    elif dataset_type == "handdb":
        dataset_dir = os.path.expanduser("~/dataset/handdb_dataset")
        import handdb_dataset as x_dataset
        from handdb_dataset import get_handdb_dataset as get_dataset
        enable_depth = False
        debug = False
    else:
        NotImplementedError("dataset_type={} is not yet".format(dataset_type))
    param = {
        "cube": np.array([200, 200, 200], dtype=np.int),
        "imsize": np.array([224, 224], dtype=np.int),
        "use_rgb": enable_rgb,
        "use_depth": enable_depth,
        "enable_x_flip": False,
        "enable_y_flip": False,
        "angle_range": range(-90,90),
        "oscillation": {
            "do_oscillate": True,
            "scale_range": np.arange(1.25, 1.27, 0.01),
            "shift_range": np.arange(-0.01, 0.02, 0.01),
        },
    }

    logger.info("get dataset")
    dataset = get_dataset(dataset_dir, param=param, debug=debug, mode="train")
    logger.info("done get dataset")
    return dataset, x_dataset, param

dataset,x_dataset,hand_param = load_dataset("fhad")

# +
current_idx=0

def vis(idx):
    global current_idx
    current_idx = idx
    color_map = x_dataset.COLOR_MAP
    keypoint_names = x_dataset.KEYPOINT_NAMES
    edges = x_dataset.EDGES
    enable_rgb = hand_param["use_rgb"] 
    enable_depth = False
    if enable_rgb and enable_depth:
        from pose.visualizations import visualize_both
        visualize_both(dataset, keypoint_names, edges, color_map)
    elif enable_rgb:
        from pose.visualizations import visualize_rgb
        visualize_rgb(dataset, keypoint_names, edges, color_map,idx)
    elif enable_depth:
        from visualizations import visualize_depth
        visualize_depth(dataset, keypoint_names, edges, color_map)
    else:
        pass

from ipywidgets import interact    

sample=np.random.choice(range(len(dataset)),100)
interact(vis,idx=sample)
# -

# # visualize transformed dataset

# +
from collections import defaultdict

from chainer.datasets import TransformDataset
from pose.models.selector import select_model
from pose.hand_dataset import common_dataset

config=defaultdict(dict)
config["model"]["name"]="ganerated"
hand_param["inH"]=224
hand_param["inW"]=224
hand_param["inC"]=3
hand_param["n_joints"]=common_dataset.NUM_KEYPOINTS
hand_param["edges"] = common_dataset.EDGES
model  = select_model(config,hand_param)
transform_dataset=TransformDataset(dataset,model.encode)

# +
print(current_idx)

rgb, hm, intermediate3d, rgb_joint = transform_dataset.get_example(current_idx)
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=fig.add_subplot(121)
ax.imshow(np.max(hm,axis=0))
ax2=fig.add_subplot(122,projection="3d")
ax2.scatter(*rgb_joint[:,::-1].transpose())
