import argparse
import configparser
import logging

logger = logging.getLogger(__name__)

import os

import cv2

cv2.setNumThreads(0)

from matplotlib import pyplot as plt

import chainer

import numpy as np
import tqdm

from pose.hand_dataset.selector import select_dataset
from pose.models.selector import select_model
from pose.hand_dataset.common_dataset import ROOT_IDX

from pose.predict import get_result_ppn


def evaluate_ppn(trained, model, dataset, hand_param):
    distances3D = []
    avg_distances3D = []
    max_distances3D = []

    distances2D = []
    avg_distances2D = []
    max_distances2D = []
    length = len(dataset)

    for idx in tqdm.tqdm(range(length)):
        example = dataset.get_example(idx)
        gt_kp_zyx = example["rgb_joint"]
        gt_kp_vu = example["rgb_camera"].zyx2vu(example["rgb_joint"])
        vmin, umin, vmax, umax = example["domain"]
        inH, inW = model.inH, model.inW
        scaleH = (vmax - vmin) / inH
        scaleW = (umax - umin) / inW
        gt_kp_zyx = gt_kp_zyx - gt_kp_zyx[ROOT_IDX]
        kp_vu, kp_zyx = get_result_ppn(model, dataset, hand_param, idx)
        kp_vu = kp_vu * np.array([scaleH, scaleW])
        gt_kp_vu = gt_kp_vu * np.array([scaleH, scaleW])
        dist_3d = np.sqrt(np.sum(np.square(kp_zyx - gt_kp_zyx), axis=1))
        dist_2d = np.sqrt(np.sum(np.square(kp_vu - gt_kp_vu), axis=1))

        distances2D.append(dist_2d)
        avg_distances2D.append(np.mean(dist_2d))
        max_distances2D.append(np.max(dist_2d))

        distances3D.append(dist_3d)
        avg_distances3D.append(np.mean(dist_3d))
        max_distances3D.append(np.max(dist_3d))

    print("2D avg distance per pixel ", np.array(avg_distances2D).mean())
    print("3D avg distance [mm] ", np.array(avg_distances3D).mean())
    print("3D average max distance [mm] ", np.array(max_distances3D).mean())

    # 2D PCK
    distances2D = np.array(distances2D)
    print(distances2D.shape)
    ps = []
    n_joints = model.n_joints
    min_threshold, max_threshold, n_plots = 0, 30, 20
    for threshold in np.linspace(min_threshold, max_threshold, n_plots):
        ratio = np.mean([np.mean(distances2D[:, j] <= threshold) for j in range(n_joints)])
        ps.append(100 * ratio)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Distance threshold / mm")
    ax.set_ylabel("PCK / %")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max_threshold)
    ax.plot(np.linspace(min_threshold, max_threshold, n_plots), ps)
    ax.grid(True, linestyle="--")
    plt.savefig(os.path.join(trained, "result", "plot_PCK_ppn2D.png"))

    # 3D PCK
    distances3D = np.array(distances3D)
    ps = []
    n_joints = model.n_joints
    min_threshold, max_threshold, n_plots = 0, 50, 15
    for threshold in np.linspace(min_threshold, max_threshold, n_plots):
        ratio = np.mean([np.mean(distances3D[:, j] <= threshold) for j in range(n_joints)])
        ps.append(100 * ratio)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Distance threshold / mm")
    ax.set_ylabel("Fraction of frames with mean below distance / %")
    ax.set_ylim(0, 100)
    ax.set_xlim(min_threshold, max_threshold)
    ax.plot(np.linspace(min_threshold, max_threshold, n_plots), ps)
    ax.grid(True, linestyle="--")
    plt.savefig(os.path.join(trained, "result", "plot_PCK_ppn3D.png"))


def main(args):
    logging.basicConfig(level=logging.INFO)

    config = configparser.ConfigParser()
    config_path = os.path.join(args.trained, "result", "config.ini")
    if not os.path.exists(config_path):
        raise Exception("config_path {} does not found".format(config_path))
    logger.info("read {}".format(config_path))
    config.read(config_path, 'UTF-8')

    logger.info("setup devices")
    chainer.global_config.autotune = True
    chainer.config.cudnn_fast_batch_normalization = True

    logger.info("> get dataset {}".format(args.mode))
    mode_dict = {
        "train": "train_set",
        "val": "val_set",
        "test": "test_set",
    }
    return_type = mode_dict[args.mode]

    dataset, hand_param = select_dataset(config, [return_type, "hand_param"])

    logger.info("> hand_param = {}".format(hand_param))
    model = select_model(config, hand_param)

    logger.info("> size of dataset is {}".format(len(dataset)))
    model_path = os.path.expanduser(os.path.join(args.trained, "result", "bestmodel.npz"))

    logger.info("> restore model")
    logger.info("> model.device = {}".format(model.device))
    chainer.serializers.load_npz(model_path, model)
    evaluate_ppn(args.trained, model, dataset, hand_param)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("trained", type=str, default="./trained")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mode", default="test")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
