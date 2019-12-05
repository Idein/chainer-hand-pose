import argparse
import configparser
import logging

logger = logging.getLogger()
import os

import numpy as np
import cv2

import chainer
import chainercv

import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from selector import select_dataset, select_model
from pose.visualizations import vis_pose
from image_utils import normalize_rgb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("trained", type=str, default="trained")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    return args


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

    dataset_type = config["dataset"]["type"]
    use_rgb = config.getboolean("dataset", "use_rgb")
    use_depth = config.getboolean("dataset", "use_depth")
    assert use_rgb
    assert use_rgb ^ use_depth, "XOR(use_rgb, use_depth) must be True"
    hand_param = select_dataset(config, return_data=["hand_param"])
    model_path = os.path.expanduser(os.path.join(args.trained, "result", "bestmodel.npz"))

    logger.info("> restore model")
    model = select_model(config, hand_param)
    logger.info("> model.device = {}".format(model.device))

    logger.info("> restore models")
    chainer.serializers.load_npz(model_path, model)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

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
            with chainer.using_config('train', False):
                heatmaps = model.pose.forward(np.expand_dims(normalize_rgb(image), axis=0))
                heatmaps = heatmaps[-1].array.squeeze()
                pts2d = []
                for i in range(len(heatmaps)):
                    hm = heatmaps[i]
                    y, x = np.unravel_index(np.argmax(hm), hm.shape)
                    pts2d.append([8 * y, 8 * x])
                joint2d = np.array(pts2d)
            color_map = hand_param["color_map"]
            keypoint_names = hand_param["keypoint_names"]
            edges = hand_param["edges"]
            color = [color_map[k] for k in keypoint_names]
            edge_color = [color_map[s, t] for s, t in edges]
            vis_pose(np.array(joint2d), edges, image, color, edge_color, ax=ax)
            fig.canvas.draw()
            buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            buf = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            # buf = cv2.resize(buf, (dW, dH))
            ax.clear()
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
