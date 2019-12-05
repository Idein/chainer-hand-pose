import argparse
import configparser
import logging

logger = logging.getLogger(__name__)

import os

import cv2

cv2.setNumThreads(0)

from pose.visualizations import vis_pose
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import chainer
import chainercv
from chainer.datasets import TransformDataset
import numpy as np
import tqdm

from pose.hand_dataset.selector import select_dataset
from pose.models.selector import select_model
from pose.hand_dataset.geometry_utils import calc_com
from pose.hand_dataset.image_utils import denormalize_rgb, normalize_rgb
from pose.hand_dataset.common_dataset import ROOT_IDX


def main(args):
    logging.basicConfig(level=logging.INFO)

    config = configparser.ConfigParser()
    config_path = os.path.join(args.trained, "pose", "config.ini")
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
    transformed_dataset = TransformDataset(dataset, model.encode)

    logger.info("> size of dataset is {}".format(len(dataset)))
    model_path = os.path.expanduser(os.path.join(args.trained, "pose", "bestmodel.npz"))

    logger.info("> restore model")
    logger.info("> model.device = {}".format(model.device))
    chainer.serializers.load_npz(model_path, model)

    if config["model"]["name"] in ["ppn", "ppn_edge"]:
        if args.evaluate:
            evaluate_ppn(model, dataset, hand_param)
        else:
            predict_ppn(model, dataset, hand_param)
    elif config["model"]["name"] in ["rhd", "hm", "orinet"]:
        predict_heatmap(model, dataset, hand_param)
    elif config["model"]["name"] == "ganerated":
        predict_ganerated(model, dataset, hand_param)
    else:
        predict_sample(model, dataset, hand_param)


def get_result_ppn(model, dataset, hand_param, idx):
    example = dataset.get_example(idx)
    image = example["rgb"]
    ret = model.predict(np.expand_dims(normalize_rgb(image), axis=0))
    if len(ret) == 7:
        resp, conf, x, y, w, h, v = ret
    else:
        resp, conf, x, y, w, h, e, v = ret
    resp = chainer.backends.cuda.to_cpu(resp.array)
    conf = chainer.backends.cuda.to_cpu(conf.array)
    w = chainer.backends.cuda.to_cpu(w.array)
    h = chainer.backends.cuda.to_cpu(h.array)
    x = chainer.backends.cuda.to_cpu(x.array)
    y = chainer.backends.cuda.to_cpu(y.array)
    v = chainer.backends.cuda.to_cpu(v.array)
    resp = np.squeeze(resp, axis=0)
    conf = np.squeeze(conf, axis=0)
    x = np.squeeze(x, axis=0)
    y = np.squeeze(y, axis=0)
    w = np.squeeze(w, axis=0)
    h = np.squeeze(h, axis=0)
    v = np.squeeze(v, axis=0)
    color_map = hand_param["color_map"]
    keypoint_names = hand_param["keypoint_names"]
    edges = hand_param["edges"]
    delta = resp * conf
    scaleH = hand_param["inH"] / model.outsize[0]
    scaleW = hand_param["inW"] / model.outsize[1]
    joint2d = {}
    grid_position = {}

    gt_3dj = example["rgb_joint"]

    for kname in keypoint_names:
        if "mcp" in kname or "root" == kname:
            i = keypoint_names.index(kname)
            u_ind = np.unravel_index(np.argmax(delta[i]), delta[i].shape)
            y_offset, x_offset = u_ind
            joint2d[kname] = [
                scaleH * (y_offset + y[i][u_ind]),
                scaleW * (x_offset + x[i][u_ind])
            ]
            grid_position[kname] = u_ind

    for f in ["thumb", "index", "middle", "ring", "little"]:
        for p, q in zip(["mcp", "pip", "dip"], ["pip", "dip", "tip"]):
            f_p = "_".join([f, p])
            f_q = "_".join([f, q])
            p_h, p_w = grid_position[f_p]
            i = keypoint_names.index(f_q)
            sz = 1 if q == "tip" else 2
            hslice = slice(max(0, p_h - sz), min(model.outsize[0], p_h + sz + 1))
            wslice = slice(max(0, p_w - sz), min(model.outsize[1], p_w + sz + 1))
            target = delta[i][hslice, wslice]
            q_h, q_w = np.unravel_index(np.argmax(target), target.shape)
            y_offset = (p_h - sz) + q_h if p_h - sz >= 0 else q_h
            x_offset = (p_w - sz) + q_w if p_w - sz >= 0 else q_w
            joint2d[f_q] = [
                scaleH * (y_offset + y[i][(y_offset, x_offset)]),
                scaleW * (x_offset + x[i][(y_offset, x_offset)])
            ]
            grid_position[f_q] = (y_offset, x_offset)
    kp_vu = np.array([joint2d[k] for k in keypoint_names])

    kp_zyx = np.zeros((len(keypoint_names), 3))
    for ei, (s, t) in enumerate(edges):
        u_ind = grid_position[keypoint_names[s]]
        orien = v[ei, :, u_ind[0], u_ind[1]]
        orien = orien / np.linalg.norm(orien)
        # use edge length from grond truth
        elen = np.sqrt(np.sum(np.square(gt_3dj[t] - gt_3dj[s])))
        # elen = 3.5 if s == 0 else 1
        kp_zyx[t] = kp_zyx[s] + orien * elen
    return kp_vu, kp_zyx


def predict_ppn(model, dataset, hand_param):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    ax3 = fig.add_subplot(122, projection="3d")
    logger.info("> use ppn")
    idx = np.random.randint(0, len(dataset))
    example = dataset.get_example(idx)
    image = example["rgb"]
    gt_kp_zyx = example["rgb_joint"]
    gt_kp_vu = example["rgb_camera"].zyx2vu(example["rgb_joint"])
    gt_kp_zyx = gt_kp_zyx - gt_kp_zyx[ROOT_IDX]
    scaleH = hand_param["inH"] / model.outsize[0]
    scaleW = hand_param["inW"] / model.outsize[1]

    kp_vu, kp_zyx = get_result_ppn(model, dataset, hand_param, idx)

    color_map = hand_param["color_map"]
    keypoint_names = hand_param["keypoint_names"]
    edges = hand_param["edges"]
    point_color = [color_map[k] for k in keypoint_names]
    edge_color = [color_map[s, t] for s, t in edges]

    kp_zyx = kp_zyx
    vis_pose(kp_vu, edges, image, point_color, edge_color, ax=ax)
    vis_pose(
        kp_zyx, edges,
        point_color=point_color,
        edge_color=edge_color,
        ax=ax3,
    )

    vis_pose(gt_kp_vu, edges, image, point_color, edge_color=[(0, 0, 0) for e in edges], ax=ax)
    vis_pose(
        gt_kp_zyx, edges,
        point_color=point_color,
        edge_color=[(0, 0, 0) for _ in edges],
        ax=ax3,
    )

    for i in range(model.outsize[0]):
        ax.plot([0, hand_param["inH"]], [i * scaleW, i * scaleW], color='w')
    for i in range(model.outsize[1]):
        ax.plot([i * scaleH, i * scaleH], [0, hand_param["inW"]], color='w')

    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.view_init(-80, -90)
    plt.show()


def evaluate_ppn(model, dataset, hand_param):
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
    plt.savefig("plot_PCK_ppn2D.png")

    # 3D PCK
    distances3D = np.array(distances3D)
    ps = []
    n_joints = model.n_joints
    min_threshold, max_threshold, n_plots = 20, 50, 15
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
    plt.savefig("plot_PCK_ppn3D.png")


def predict_ganerated(model, dataset, hand_param):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    ax3 = fig.add_subplot(122, projection="3d")

    keypoint_names = hand_param["keypoint_names"]
    edges = hand_param["edges"]
    color_map = hand_param["color_map"]

    idx = np.random.randint(0, len(dataset))
    example = dataset.get_example(idx)
    inp = example["rgb"] / 255
    with chainer.using_config('train', False):
        heatmaps = model.predict(np.expand_dims(inp, axis=0))
        heatmaps = heatmaps[-1].array.squeeze()
        pts2d = []
        for i in range(len(heatmaps)):
            hm = heatmaps[i]
            logger.info(hm.shape)
            y, x = np.unravel_index(np.argmax(hm), hm.shape)
            print(y, x)
            y = hand_param["inH"] / hm.shape[0] * y
            x = hand_param["inW"] / hm.shape[1] * x
            pts2d.append([y, x])
        pts2d = np.array(pts2d)
        point_color = [color_map[k] for k in keypoint_names]
        edge_color = [color_map[(s, t)] for (s, t) in edges]
        vis_pose(
            pts2d, edges,
            img=example["rgb"],
            point_color=point_color,
            edge_color=edge_color, ax=ax
        )
        # vis_pose(
        #    pred_canonical_joint, edges,
        #    point_color=point_color,
        #    edge_color=edge_color,
        #    ax=ax3
        # )
    plt.show()


def predict_heatmap(model, dataset, hand_param):
    from model_rhd import variable_rodrigues
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    ax3 = fig.add_subplot(122, projection="3d")

    keypoint_names = hand_param["keypoint_names"]
    edges = hand_param["edges"]
    color_map = hand_param["color_map"]

    idx = np.random.randint(0, len(dataset))
    example = dataset.get_example(idx)
    inp = example["rgb"] / 255
    with chainer.using_config('train', False):
        heatmaps = model.pose.forward(np.expand_dims(inp, axis=0))
        # pred_canonical_joint = model.prior(heatmaps).reshape(-1, 3)
        # pred_R = variable_rodrigues(model.rot(heatmaps))
        heatmaps = heatmaps[-1].array.squeeze()
        # pred_canonical_joint = pred_canonical_joint.array
        # pred_R = pred_R.array
        pts2d = []
        for i in range(len(heatmaps)):
            hm = heatmaps[i]
            y, x = np.unravel_index(np.argmax(hm), hm.shape)
            pts2d.append([hand_param["inH"] / hm.shape[0] * y, hand_param["inW"] / hm.shape[1] * x])
        pts2d = np.array(pts2d)
        point_color = [color_map[k] for k in keypoint_names]
        edge_color = [color_map[(s, t)] for (s, t) in edges]
        vis_pose(
            pts2d, edges,
            img=example["rgb"],
            point_color=point_color,
            edge_color=edge_color, ax=ax
        )
        # vis_pose(
        #    pred_canonical_joint, edges,
        #    point_color=point_color,
        #    edge_color=edge_color,
        #    ax=ax3
        # )
    plt.show()


def evaluate(model, dataset, hand_param, debug):
    transformed_dataset = TransformDataset(dataset, model.encode)
    avg_distances = []
    max_distances = []
    length = len(transformed_dataset) if not debug else 10

    for idx in tqdm.tqdm(range(length)):
        image, gt_2dj, gt_3dj = transformed_dataset.get_example(idx)
        example = dataset.get_example(idx)
        pred_j = model.predict(np.array([image], dtype=np.float32))
        with chainer.using_config('train', False):
            loss = model.forward(
                np.expand_dims(image, axis=0),
                np.expand_dims(gt_3dj, axis=0),
                np.expand_dims(gt_2dj, axis=0),
            )
        pred_j = pred_j.array.reshape(hand_param["n_joints"], -1)
        dim = pred_j.shape[-1]
        if dim == 5:
            pred_3d = pred_j[:, :3]
            pred_2d = pred_j[:, 3:]
        else:
            pred_3d = pred_j

        logger.debug("> {}".format(pred_j))
        logger.debug("> loss {}".format(loss))
        logger.debug("> visualize pred_joint")

        z_half = hand_param["cube"][0] / 2
        pred_3d = z_half * pred_3d
        gt_3dj = example["rgb_joint"] if hand_param["use_rgb"] else example["depth_joint"]
        gt_3dj = gt_3dj - calc_com(gt_3dj)
        dist = np.sqrt(np.sum(np.square(pred_3d - gt_3dj), axis=1))
        avg_dist = np.mean(dist)
        max_dist = np.max(dist)
        avg_distances.append(avg_dist)
        max_distances.append(max_dist)

    print(np.array(avg_distances).mean())
    max_distances = np.array(max_distances)
    ps = []
    max_threshold = 80
    for threshold in range(3, max_threshold):
        oks = np.sum(max_distances <= threshold)
        percent = 100 * (oks / len(max_distances))
        ps.append(percent)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Distance threshold / mm")
    ax.set_ylabel("Fraction of frames iwth mean below distance / %")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max_threshold)
    ax.plot(ps)
    ax.grid(True, linestyle="--")
    plt.savefig("plot.png")


def predict_sample(model, dataset, hand_param):
    transformed_dataset = TransformDataset(dataset, model.encode)
    idx = np.random.randint(0, len(transformed_dataset))
    image, gt_2dj, gt_3dj = transformed_dataset.get_example(idx)
    example = dataset.get_example(idx)

    vis_vu = gt_2dj * np.array([[hand_param["inH"], hand_param["inW"]]])
    pred_j = model.predict(np.array([image], dtype=np.float32))
    with chainer.using_config('train', False):
        loss = model.forward(
            np.expand_dims(image, axis=0),
            np.expand_dims(gt_3dj, axis=0),
            np.expand_dims(gt_2dj, axis=0),
        )
    pred_j = pred_j.array.reshape(hand_param["n_joints"], -1)
    dim = pred_j.shape[-1]
    if dim == 5:
        pred_3d = pred_j[:, :3]
        pred_2d = pred_j[:, 3:]
        pred_2d = pred_2d * np.array([[hand_param["inH"], hand_param["inW"]]])
    else:
        pred_3d = pred_j
    logger.info("> {}".format(pred_j))
    logger.info("> loss {}".format(loss))
    logger.info("> visualize pred_joint")
    plot_direction = "horizontal"
    if plot_direction == "horizontal":
        space = (1, 2)
        figsize = (10, 5)
    else:
        space = (2, 1)
        figsize = (5, 10)
    z_half = hand_param["cube"][0] / 2
    pred_3d = z_half * pred_3d
    gt_3dj = example["rgb_joint"] if hand_param["use_rgb"] else example["depth_joint"]
    gt_3dj = gt_3dj - calc_com(gt_3dj)
    distance = np.sqrt(np.sum(np.square(pred_3d - gt_3dj), axis=1)).mean()
    logger.info("> mean distance {:0.2f}".format(distance))
    fig = plt.figure(figsize=figsize)
    fig.suptitle("mean distance = {:0.2f}".format(distance))
    ax1 = fig.add_subplot(*space, 1)
    ax1.set_title("result 2D")
    ax2 = fig.add_subplot(*space, 2, projection="3d")
    ax2.set_title("result 3D")
    color_map = hand_param["color_map"]
    keypoint_names = hand_param["keypoint_names"]
    edges = hand_param["edges"]
    color = [color_map[k] for k in keypoint_names]
    pred_color = [color_map[s, t] for s, t in edges]
    gt2_color = [[255, 255, 255] for k in keypoint_names]
    gt3_color = [[50, 50, 50] for k in keypoint_names]
    if hand_param["use_rgb"]:
        image = denormalize_rgb(image)
        chainercv.visualizations.vis_image(image, ax=ax1)
    else:
        image = image.squeeze()
        ax1.imshow(image, cmap="gray")
    vis_pose(vis_vu, edges, point_color=color,
             edge_color=gt2_color, ax=ax1)
    vis_pose(pred_2d, edges, point_color=color,
             edge_color=pred_color, ax=ax1)
    vis_pose(gt_3dj, edges, point_color=color, edge_color=gt3_color, ax=ax2)
    if dim != 2:
        vis_pose(pred_3d, edges, point_color=color, edge_color=pred_color, ax=ax2)
    # set layout
    for ax in [ax2]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(-65, -90)
    # show
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained", type=str, default="./trained")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mode", default="test")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
