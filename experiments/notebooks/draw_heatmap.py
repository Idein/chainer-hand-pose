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

import math
from scipy.stats import norm
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# +
EPS = 1e-6


def draw_gaussian(H, W, cx, cy, sigma):
    xx, yy = np.meshgrid(np.arange(0, W), np.arange(0, H))
    half_sigma_square = 2 * sigma * sigma
    heat = np.exp((-(xx - cx) ** 2 - (yy - cy) ** 2) / half_sigma_square)
    # normalize heat takes value in the interval [0,1] \in \mathbb{R}
    heat[heat < 0.1] = 0
    return heat


# +
H,W = 256,256
heatmaps = np.empty((10,H,W))

for i in range(len(heatmaps)):
    cx = np.random.randint(H)
    cy = np.random.randint(W)
    heatmaps[i]=draw_gaussian(H,W, cx, cy, sigma=5)
heatmaps=heatmaps.astype(np.float32)
# -

# %matplotlib notebook 
plt.imshow(np.sum(heatmaps,axis=0),vmin=0,vmax=1)

# +
EPS = 1e-6


def draw_gaussian(H, W, cx, cy, sigma):
    xx, yy = np.meshgrid(np.arange(0, W), np.arange(0, H))
    half_sigma_square = 2 * sigma * sigma
    heat = np.exp((-(xx - cx) ** 2 - (yy - cy) ** 2) / half_sigma_square)
    # normalize heat takes value in the interval [0,1] \in \mathbb{R}
    heat = heat / (heat.max() + EPS)
    heat[heat < 0.1] = 0
    return heat


# +
from chainer.backends.cuda import get_array_module
import chainer
import chainer.functions as F

xp = np
def variable_heatmap(H, W, cx, cy, sigma=5):
    xp = get_array_module(cx)
    xx, yy = xp.meshgrid(xp.arange(0, W), np.arange(0, H))
    xx = xx.astype(xp.float32)
    yy = yy.astype(xp.float32)
    n_joints = cx.shape[0]
    numX = F.stack(n_joints * [xx]) - cx[:, None, None]
    numY = F.stack(n_joints * [yy]) - cy[:, None, None]
    hmap = F.exp((-numX**2 - numY**2) / 2 / sigma / sigma)
    return hmap



# +
from ipywidgets import interact
# %matplotlib inline

pts = chainer.Variable(np.array(
    [[100, 200,90],
     [150, 45,56]], 
    np.float32))
cx, cy = pts
print(cx,cy)
hmaps = variable_heatmap(200, 200, cx, cy, 8)


def vis_hm(idx):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    hm = hmaps[idx].array
    ax.imshow(hm, vmin=0, vmax=1)


interact(vis_hm, idx=range(len(cx)))
