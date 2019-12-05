import argparse
import configparser
import logging

logger = logging.getLogger()
import os

import cv2

import chainer
import chainercv

import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from model import HandPoseNetwork
from selector import select_dataset
from pose.visualizations import vis_pose
from image_utils import normalize_rgb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained", type=str, default="trained")
    parser.add_argument("--camera", type=int, default=0)
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

    path = os.path.expanduser(os.path.join(args.trained, "src", "config.ini"))
    logger.info("read {}".format(path))
    config.read(path, 'UTF-8')

    logger.info("setup devices")
    chainer.global_config.autotune = True
    chainer.config.cudnn_fast_batch_normalization = True

    dataset_type = config["dataset"]["type"]
    use_rgb = config.getboolean("dataset", "use_rgb")
    use_depth = config.getboolean("dataset", "use_depth")
    assert use_rgb
    assert use_rgb ^ use_depth, "XOR(use_rgb, use_depth) must be True"
    hand_param = select_dataset(config, return_data=["hand_param"])
    model_path = os.path.expanduser(os.path.join(args.trained, "bestmodel.npz"))

    logger.info("> restore model")
    model = create_model(config, hand_param)
    logger.info("> model.device = {}".format(model.device))
    chainer.serializers.load_npz(model_path, model)

    plot_direction = "horizontal"
    if plot_direction == "horizontal":
        space = (1, 2)
        figsize = (10, 5)
    else:
        space = (2, 1)
        figsize = (5, 10)

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(*space, 1)
    ax3 = fig.add_subplot(*space, 2, projection="3d")

    color_map = hand_param["color_map"]
    color = [color_map[k] for k in hand_param["keypoint_names"]]
    edge_color = [color_map[s, t] for s, t in hand_param["edges"]]
    pred_color = [[255, 255, 255] for k in hand_param["keypoint_names"]]

    cap = cv2.VideoCapture(args.camera)
    if cap.isOpened() is False:
        print('Error opening video stream or file')
        exit(1)

    try:
        while cap.isOpened():
            # Wait for a coherent pair of frames: depth and color
            ret_val, image = cap.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.transpose(2, 0, 1).astype(np.float32)
            _, cH, cW = image.shape
            sz = min(cH, cW)
            image = chainercv.transforms.center_crop(image, (sz, sz))
            image = chainercv.transforms.resize(image, (hand_param["inH"], hand_param["inW"]))
            pred_j = model.predict(np.expand_dims(normalize_rgb(image), axis=0))
            pred_j = pred_j.array.reshape(hand_param["n_joints"], -1)
            dim = pred_j.shape[-1]
            if dim == 5:
                pred_3d = pred_j[:, :3]
                pred_2d = pred_j[:, 3:]
                pred_2d = pred_2d * np.array([[hand_param["inH"], hand_param["inW"]]])
            else:
                pred_3d = pred_j

            vis_pose(pred_2d, hand_param["edges"], img=image,
                     point_color=color, edge_color=pred_color, ax=ax1)
            if dim != 2:
                vis_pose(pred_3d, hand_param["edges"], point_color=color, edge_color=edge_color, ax=ax3)
            # set layout
            for ax in [ax3]:
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                ax.view_init(-65, -90)

            fig.canvas.draw()
            buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            buf = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            # buf = cv2.resize(buf, (dW, dH))
            ax1.clear()
            ax3.clear()

            images = np.hstack((buf,))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            if cv2.waitKey(1) == 27:
                break
            cv2.waitKey(1)
    finally:
        print("Exit")


if __name__ == '__main__':
    main()
