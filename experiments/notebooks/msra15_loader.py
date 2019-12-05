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

# +
import os
import struct

from matplotlib import pyplot as plt
import numpy as np
# -

# # Read binary file using Python
# While the depth image is 320x240, the valid hand region is usually much smaller. To save space, each *.bin file only stores the bounding box of the hand region. Specifically, each bin file starts with 6 unsigned int: img_width img_height left top right bottom. [left, right) and [top, bottom) is the bounding box coordinate in this depth image. The bin file then stores all the depth pixel values in the bounding box in row scanning order, which are  (right - left) * (bottom - top) floats. The unit is millimeters. The bin file is binary and needs to be opened with std::ios::binary flag.

# # Use struct module 
# - https://docs.python.org/2/library/struct.html

rootdir=os.path.expanduser("~/dataset/cvpr15_MSRAHandGestureDB")
folder1 = "P0"
number = "1"
frameid = "000000"
binary_name = os.path.join(rootdir, folder1, number, "_".join([frameid, "depth.bin"]))
assert_image_name = os.path.join(rootdir, folder1,number,"_".join([frameid,"depth.jpg"]))
jointfile = os.path.join(rootdir, folder1, number, "joint.txt")

with open(binary_name, mode='rb') as f:
    data = f.read()

num_unsigned_int = 6
sizeof_I = 4
fmt = "{}I".format(num_unsigned_int)
img_width, img_height, left, top, right, bottom = struct.unpack(
    fmt, data[:sizeof_I*num_unsigned_int])
num_floats = (right - left)*(bottom - top)
sizeof_f = 4
fmt = "{}f".format(num_floats)
depth_image=struct.unpack(fmt, data[sizeof_f*num_unsigned_int:])

img_width, img_height, left,top,right,bottom

depth_image=np.array(depth_image).reshape((bottom-top,right-left))
print(depth_image.max(),depth_image.min())

plt.imshow(depth_image)

from PIL import Image
cropped = Image.open(assert_image_name).crop((left, top, right, bottom))
plt.imshow(cropped)

# # Read binary using NumPy

# +
fsize = os.path.getsize(binary_name)
sizeof_I = 4
sizeof_f = 4
count = (fsize-6*sizeof_I)//sizeof_f
record_dtype = np.dtype(
    [
        ('imginfo', '6I'),
        ('depth_array', '{}f'.format(count))
    ]
)

mock = np.fromfile(binary_name, dtype=record_dtype)
# -

img_width, img_height, left,top,right,bottom=mock["imginfo"].ravel()

dm=mock["depth_array"].reshape((bottom-top,right-left))

print(dm.max,dm.min())
plt.imshow(dm)

assert (depth_image==dm).all()

# # Read joint
#
# joint.txt file stores 500 frames x 21 hand joints per frame. Each line has 3 * 21 = 63 floats for 21 3D points in (x, y, z) coordinates. The 21 hand joints are: wrist, index_mcp, index_pip, index_dip, index_tip, middle_mcp, middle_pip, middle_dip, middle_tip, ring_mcp, ring_pip, ring_dip, ring_tip, little_mcp, little_pip, little_dip, little_tip, thumb_mcp, thumb_pip, thumb_dip, thumb_tip.
#

anns=np.loadtxt(
    jointfile,
    skiprows=1 # ignore first row
)

# +
import itertools


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


KEYPOINT_NAMES = [
    "wrist",
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
    "thumb_mcp",
    "thumb_pip",
    "thumb_dip",
    "thumb_tip",
]

# (R,G,B)
BASE_COLOR = {
    "index": (0, 255, 0),
    "middle": (0, 0, 255),
    "ring": (255, 0, 255),
    "little": (255, 255, 0),
    "thumb": (255, 0, 0),
    "wrist": (50,50,50),
}

# convert tuple to numpy array
BASE_COLOR = {k: np.array(v) for k, v in BASE_COLOR.items()}

COLOR_MAP = {"wrist": BASE_COLOR["wrist"]}
EDGES_BY_NAME = []

for f in ["index", "middle", "ring", "little", "thumb"]:
    for p, q in pairwise(["wrist", "mcp", "pip", "dip", "tip"]):
        color = BASE_COLOR[f]
        if p != "wrist":
            p = "_".join([f, p])
        q = "_".join([f, q])
        COLOR_MAP[p, q] = color
        COLOR_MAP[q] = color
        EDGES_BY_NAME.append([p, q])

EDGES = [[KEYPOINT_NAMES.index(s), KEYPOINT_NAMES.index(t)]
         for s, t in EDGES_BY_NAME]

for s, t in EDGES_BY_NAME:
    COLOR_MAP[
        KEYPOINT_NAMES.index(s),
        KEYPOINT_NAMES.index(t)
    ] = COLOR_MAP[s, t]
    COLOR_MAP[KEYPOINT_NAMES.index(s)] = COLOR_MAP[s]
    COLOR_MAP[KEYPOINT_NAMES.index(t)] = COLOR_MAP[t]

# +
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib notebook
fig=plt.figure()
ax=fig.add_subplot(111,projection="3d")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

joints = anns[0].reshape(-1,3)
joints = joints-joints[KEYPOINT_NAMES.index("wrist")]
xs= joints[:,0]
ys= joints[:,1]
zs= joints[:,2]
colors = [COLOR_MAP[k]/255. for k in KEYPOINT_NAMES]
ax.scatter(xs,ys,zs,color=colors)

for s,t in EDGES:
    sx = xs[s]
    sy = ys[s]
    sz = zs[s]
    tx = xs[t]
    ty = ys[t]
    tz = zs[t]
    color = COLOR_MAP[s,t]/255.
    ax.plot([sx,tx],[sy,ty],[sz,tz], color=color)
# -

# # Project joint 2d surface
#
# In total 9 subjects' right hands are captured using Intel's Creative Interactive Gesture Camera. Each subject has 17 gestures captured and there are 500 frames for each gesture.
#
# The camera intrinsic parameters are: principle point = image center(160, 120), focal length = 241.42.

# +
EPS = 1e-8
fx = fy = 241.42
cx, cy = (160, 120)

intrinsic_matrix = [
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
]

joints = anns[0].reshape(-1, 3)

fig, ax = plt.subplots()
img_width, img_height, left, top, right, bottom
img = np.zeros((img_height, img_width), dtype=np.float32)
img[top:bottom, left:right] = dm
ax.imshow(img)

for s, t in EDGES:
    su, sv, _ = np.dot(intrinsic_matrix, joints[s]/(joints[s][-1]+EPS))
    tu, tv, _ = np.dot(intrinsic_matrix, joints[t]/(joints[t][-1]+EPS))
    color = COLOR_MAP[s,t]/255.
    # flip axis
    su = img_width - su
    tu = img_width - tu
    ax.scatter(tu,tv, color=color)
    ax.plot([su,tu],[sv,tv],color=color)
else:
    #finally draw wrist point
    wi = KEYPOINT_NAMES.index("wrist")
    color = COLOR_MAP[wi]/255.
    wu, wv, _ = np.dot(intrinsic_matrix, joints[wi]/(joints[wi][-1]+EPS))
    wu = img_width - wu
    ax.scatter(wu,wv,color=color)
