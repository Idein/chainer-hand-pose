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

import cv2
import numpy as np
from matplotlib import pyplot as plt


def simulate(inH, inW):
    img = 255*np.random.random((480, 680, 3)).astype(np.uint8)
    H, W, _ = img.shape
    aspect = min(H, W)/max(inH, inW)
    print("H, W", H, W)
    cropH = int(aspect*inH)
    cropW = int(aspect*inW)
    print("cropH, cropW", cropH, cropW)
    y_offset = int(round(H - cropH) / 2.)
    x_offset = int(round(W - cropW) / 2.)
    print("y,x",y_offset,x_offset)
    y_slice = slice(y_offset, y_offset+cropH)
    x_slice = slice(x_offset, x_offset+cropW)
    img = img[y_slice, x_slice, :]
    cH, cW, _ = img.shape
    print(cH, cW, cH/cW)
    print(inH, inW, inH/inW)


simulate(300, 200)

simulate(200,300)
