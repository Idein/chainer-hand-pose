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

import numpy as np
from matplotlib import pyplot as plt
import chainercv
from chainercv.visualizations import vis_image
import skimage


def relu(x):
    return max(x, 0)


def intersection(bbox0, bbox1):
    y0m, x0m, y0M, x0M = bbox0
    y1m, x1m, y1M, x1M = bbox1
    w = relu(min(x0M, x1M)-max(x0m, x1m))
    h = relu(min(y0M, y1M)-max(y0m, y1m))
    return h, w


def new_crop(image, domain, fill=0):
    C, H, W = image.shape
    bbox0 = domain
    y0m, x0m, y0M, x0M = bbox0
    bbox1 = [0, 0, H, W]
    y1m, x1m, y1M, x1M = bbox1
    outH = y0M - y0m
    outW = x0M - x0m
    h, w = intersection(bbox0, bbox1)
    if h * w == 0:
        new_canvas = np.empty((C, outH, outW), dtype=image.dtype)
        new_canvas[:] = np.array(fill).reshape((-1, 1, 1))
        param = {
            "y_offset": -y0m,
            "x_offset": -x0m,
        }
        return new_canvas, param
    else:
        new_canvas = np.empty((C, outH, outW), dtype=image.dtype)
        new_canvas[:] = np.array(fill).reshape((-1, 1, 1))
        sx, sy = max(x0m, x1m), max(y0m, y1m)
        y_slice, x_slice = slice(sy, sy + h), slice(sx, sx + w)
        cropped = image[:, y_slice, x_slice]
        y_offset = relu(y1m - y0m)
        x_offset = relu(x1m - x0m)
        y_canvas_slice = slice(y_offset, y_offset + h)
        x_canvas_slice = slice(x_offset, x_offset + w)
        new_canvas[:, y_canvas_slice, x_canvas_slice] = cropped
        y_offset = -sy+y_offset
        x_offset = -sx+x_offset
        param = {
            "y_offset": y_offset,
            "x_offset": x_offset,
        }
        return new_canvas, param


image = skimage.data.astronaut().transpose(2, 0, 1)
point=np.array([[
    [100,195],
    [100,250],
]])
chainercv.visualizations.vis_point(image,point)

import itertools
image = skimage.data.astronaut().transpose(2, 0, 1)
_, imH, imW = image.shape
x_left = -90
x_middle = imW//2-100
x_right = imW+90
y_left = -90
y_middle = imH//2
y_right = imH+90
xms = np.array([x_left, x_middle, x_right])
xMs = xms+200
yms = np.array([y_left, y_middle, y_right])
yMs = yms+200
testset = []
for ym, xm, yM, xM in itertools.product(*[yms, xms, yMs, xMs]):
    if ym < yM and xm < xM:
        domain = [ym, xm, yM, xM]
        print(domain)
        cropped, param = new_crop(image,domain)
        print(param)
        t=chainercv.transforms.translate_point(point,x_offset=param["x_offset"],y_offset=param["y_offset"])
        chainercv.visualizations.vis_point(cropped,t)
