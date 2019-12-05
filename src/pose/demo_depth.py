# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import argparse
import configparser
import logging
logger = logging.getLogger()
import os

import pyrealsense2 as rs
import numpy as np
import cv2

import chainer
import chainercv
from chainer.datasets import TransformDataset

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from pose.visualizations import vis_pose
from image_utils import normalize_depth
#from image_utils import VALID_MIN, VALID_MAX
from selector import select_dataset, select_model
from pose.utils import parse_kwargs, parse_imsize, parse_cube

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)


import numpy as np
from scipy import stats

VALID_MIN = 30
VALID_MAX = 1500


def convert_depth_to_uvd(depth):
    if depth.ndim == 2:
        d = np.expand_dims(depth, axis=0)
    d = depth
    _, H, W = d.shape
    uv = np.meshgrid(range(W), range(H))
    uvd = np.concatenate([uv, d], axis=0)
    return uvd


def define_background(depth):
    valid_loc = np.logical_and(VALID_MIN <= depth, depth <= VALID_MAX)
    # define background as most frequently occurring number i.e. mode
    valid_depth = depth[valid_loc]
    mode_val, mode_num = stats.mode(valid_depth.ravel())
    background = mode_val.squeeze()
    return background


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("trained", type=str, default="~/tmp/trained")
    args = parser.parse_args()
    return args


def create_model(config, hand_param):
    model_name = config["model"]["name"]
    logger.info("use {}".format(model_name))
    if model_name == "mv2":
        model_param = {
            "width_multiplier": config.getfloat(model_name, "width_multiplier"),
        }
    elif model_name == "resnet":
        model_param = {
            "n_layers": config.getint(model_name, "n_layers")
        }
    elif model_name == "deep_prior":
        model_param = {}
    model = HandPoseNetwork(hand_param, model_name, model_param)
    return model


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    config = configparser.ConfigParser()

    path = os.path.expanduser(os.path.join(args.trained, "result", "config.ini"))
    logger.info("read {}".format(path))
    config.read(path, 'UTF-8')

    logger.info("setup devices")
    chainer.global_config.autotune = True
    chainer.config.cudnn_fast_batch_normalization = True

    dataset_type = config.get("dataset", "type")
    use_rgb = config.getboolean("dataset", "use_rgb")
    use_depth = config.getboolean("dataset", "use_depth")
    assert use_rgb ^ use_depth, "XOR(use_rgb, use_depth) must be True"
    cube = parse_cube(config[dataset_type]["cube"], style="DHW")
    hand_param = select_dataset(config, return_data=["hand_param"])
    model_path = os.path.expanduser(os.path.join(args.trained, "result", "bestmodel.npz"))

    logger.info("> restore model")
    model = select_model(config, hand_param)
    print(model)
    logger.info("> model.device = {}".format(model.device))
    chainer.serializers.load_npz(model_path, model)

    plot_direction = "horizontal"
    if plot_direction == "horizontal":
        space = (1, 3)
        figsize = (15, 5)
    else:
        space = (3, 1)
        figsize = (5, 15)

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(*space, 1)
    ax2 = fig.add_subplot(*space, 2)
    ax3 = fig.add_subplot(*space, 3, projection="3d")

    color_map = hand_param["color_map"]
    color = [color_map[k] for k in hand_param["keypoint_names"]]
    edge_color = [color_map[s, t] for s, t in hand_param["edges"]]
    pred_color = [[255, 255, 255] for k in hand_param["keypoint_names"]]

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            #color_frame = frames.get_color_frame()
            if not depth_frame:
                continue
            # if not color_frame:
            #    continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            #color_image = np.asanyarray(color_frame.get_data())
            logger.info("> depth_image {} {} {}".format(depth_image.min(), depth_image.max(), depth_image.dtype))
            dH, dW = depth_image.shape
            #cH, cW, _ = color_image.shape

            size = 448  # hard coded
            dhslice = slice(dH // 2 - size // 2, dH // 2 - size // 2 + size)
            dwslice = slice(dW // 2 - size // 2, dW // 2 - size // 2 + size)
            depth_image = depth_image[dhslice, dwslice]

            inp = chainercv.transforms.center_crop(
                np.expand_dims(depth_image, axis=0),
                (224, 224),
                copy=True,
            ).astype(np.float32)
            #inp = chainercv.transforms.resize(inp, (224, 224))
            _, inpH, inpW = inp.shape
            z_com = inp[0, inpH // 2, inpW // 2]

            logger.info("> com size {} {}".format(z_com, hand_param["cube"][0]))

            inp = normalize_depth(
                inp,
                z_com=z_com,
                z_size=hand_param["cube"][0],
            )
            logger.info("> normalized depth {} {} {}".format(inp.min(), inp[0, inpH // 2, inpW // 2], inp.max()))

            inp = chainercv.transforms.resize(inp, (hand_param["inH"], hand_param["inW"]))

            ax2.imshow(inp.squeeze(), cmap="gray", vmin=-1, vmax=1)
            pred_j = model.predict(np.expand_dims(inp, axis=0).astype(np.float32))

            pred_j = pred_j.array.reshape(hand_param["n_joints"], -1)
            dim = pred_j.shape[-1]
            if dim == 5:
                pred_3d = pred_j[:, :3]
                pred_2d = pred_j[:, 3:]
                pred_2d = pred_2d * np.array([[hand_param["inH"], hand_param["inW"]]])
            else:
                pred_3d = pred_j

            ax1.imshow(np.asarray(depth_image), cmap="gray")
            vis_pose(pred_2d, hand_param["edges"],
                     point_color=color, edge_color=pred_color, ax=ax2)
            if dim != 2:
                vis_pose(pred_3d, hand_param["edges"], point_color=color, edge_color=edge_color, ax=ax3)
            # set layout
            for ax in [ax3]:
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                ax.view_init(-65, -90)
            ax2.set_xlim(0, hand_param["inW"])
            ax2.set_ylim(0, hand_param["inH"])
            ax2.invert_yaxis()
            fig.canvas.draw()
            buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            buf = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            # buf = cv2.resize(buf, (dW, dH))
            ax1.clear()
            ax2.clear()
            ax3.clear()

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            # depth_colormap = cv2.applyColorMap(
            #    cv2.convertScaleAbs(depth_image, alpha=0.03),
            #    cv2.COLORMAP_JET
            # )

            # Stack both images horizontally
            # images = np.hstack((color_image, depth_colormap))
            # images = np.hstack((buf, depth_colormap))
            images = np.hstack((buf,))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            if cv2.waitKey(1) == 27:
                break
            cv2.waitKey(1)
    finally:
        # Stop streaming
        pipeline.stop()


if __name__ == '__main__':
    main()
