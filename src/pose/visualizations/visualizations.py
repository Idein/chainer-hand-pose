import logging

logger = logging.getLogger(__name__)

import random

import chainercv
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA

from pose.hand_dataset.geometry_utils import normalize_joint_zyx
from pose.hand_dataset.image_utils import normalize_depth

# Decimal Code (R,G,B)
BASE_COLOR = {
    "RED": (255, 0, 0),
    "GREEN": (0, 255, 0),
    "BLUE": (0, 0, 255),
    "YELLOW": (255, 255, 0),
    "CYAN": (0, 255, 255),
    "MAGENTA": (255, 0, 255),
}


def vis_image(img, ax=None):
    """
    extend chainercv.visualizations.vis_image
    """
    C, H, W = img.shape
    if C == 1:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        # remove channnel dimension
        ax.imshow(img.squeeze())
    else:
        ax = chainercv.visualizations.vis_image(img, ax)
    return ax


def preprocess(point, ax, img):
    input_point = np.asarray(point)

    if input_point.ndim == 2:
        input_point = np.expand_dims(point, axis=0)
    H, W = None, None
    if ax is None:
        fig = plt.figure()
        if input_point.shape[-1] == 3:
            ax = fig.add_subplot(1, 1, 1, projection="3d")
        else:
            ax = fig.add_subplot(1, 1, 1)
    if img is not None:
        ax = vis_image(img, ax=ax)
        _, H, W = img.shape
    return input_point, ax, H, W


def vis_point(point, img=None, color=None, ax=None):
    """
    Visualize points in an image, customized to our purpose.
    Base implementation is taken from chainercv.visualizations.vis_image
    """
    point, ax, H, W = preprocess(point, ax, img)
    n_inst = len(point)
    c = np.asarray(color) / 255. if color is not None else None
    for i in range(n_inst):
        # note that the shape of `point[i]` is (K,N) and the format of one is (y, x), (z,y,x).
        # (K, N) -> (N, K)
        pts = point[i].transpose()  # (K,N) -> (N,K)
        # resort coordinate order : yx -> xy or zyx -> xyz
        pts = pts[::-1]
        ax.scatter(*pts, c=c)
    if W is not None:
        ax.set_xlim(left=0, right=W)
    if H is not None:
        ax.set_ylim(bottom=H - 1, top=0)
    return ax


def vis_edge(point, indices, img=None, color=None, ax=None):
    """
    Visualize edges in an image
    """
    point, ax, H, W = preprocess(point, ax, img)
    n_inst = len(point)
    if color is not None:
        color = np.asarray(color) / 255.
    else:
        color = [None] * len(indices)
    for i in range(n_inst):
        # note that the shape of `point[i]` is (K,N) and the format of one is (y, x) or (z,y,x).
        pts = point[i]
        for ((s, t), c) in zip(indices, color):
            # Select point which consists edge. It is a pair or point (start, target).
            # Note that [::-1] does resort coordinate order: yx -> xy or zyx -> xyz
            edge = pts[[s, t]].transpose()
            edge = edge[::-1]
            ax.plot(*edge, c=c)
    if W is not None:
        ax.set_xlim(left=0, right=W)
    if H is not None:
        ax.set_ylim(bottom=H - 1, top=0)
    return ax


def vis_pose(point, indices, img=None, point_color=None, edge_color=None, ax=None):
    ax = vis_point(point, img=img, color=point_color, ax=ax)
    vis_edge(point, indices, img=img, color=edge_color, ax=ax)


def visualize_both(dataset, keypoint_names, edges, color_map, normalize=False):
    import random
    idx = random.randint(0, len(dataset) - 1)
    logger.info("get example")
    example = dataset.get_example(idx)
    logger.info("Done get example")
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223, projection="3d")
    ax4 = fig.add_subplot(224, projection="3d")
    color = [color_map[k] for k in keypoint_names]
    edge_color = [color_map[s, t] for s, t in edges]

    depth = example["depth"].astype(np.float32)
    depth_joint = example["depth_joint"]
    depth_camera = example["depth_camera"]
    depth_vu, depth_z = depth_camera.zyx2vu(depth_joint, return_z=True)
    z_size = example["param"]["z_size"]
    if normalize:
        depth = normalize_depth(depth, z_com=depth_z.mean(), z_size=z_size)
        depth_joint = normalize_joint_zyx(depth_joint, depth_camera, z_size)

    rgb = example["rgb"]
    rgb_joint = example["rgb_joint"]
    rgb_camera = example["rgb_camera"]
    rgb_vu = rgb_camera.zyx2vu(rgb_joint)
    rgb_joint = normalize_joint_zyx(rgb_joint, rgb_camera, z_size)

    print(example["param"])

    vis_point(rgb_vu, img=rgb, color=color, ax=ax1)
    vis_edge(rgb_vu, indices=edges, color=edge_color, ax=ax1)

    vis_point(rgb_joint, color=color, ax=ax3)
    vis_edge(rgb_joint, indices=edges, color=edge_color, ax=ax3)

    vis_point(depth_vu, img=depth, color=color, ax=ax2)
    vis_edge(depth_vu, indices=edges, color=edge_color, ax=ax2)

    vis_point(depth_joint, color=color, ax=ax4)
    vis_edge(depth_joint, indices=edges, color=edge_color, ax=ax4)

    for ax in [ax3, ax4]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(-65, -90)

    plt.savefig("output.png")
    plt.show()


def visualize_rgb(dataset, keypoint_names, edges, color_map, idx=None):
    import random
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)
    logger.info("get example")
    example = dataset.get_example(idx)
    logger.info("Done get example")
    fig = plt.figure(figsize=(5, 10))
    ax1 = fig.add_subplot(211)
    ax3 = fig.add_subplot(212, projection="3d")
    color = [color_map[k] for k in keypoint_names]
    edge_color = [color_map[s, t] for s, t in edges]

    rgb = example["rgb"]
    rgb_joint = example["rgb_joint"]
    rgb_camera = example["rgb_camera"]
    rgb_vu = rgb_camera.zyx2vu(rgb_joint)

    vis_point(rgb_vu, img=rgb, color=color, ax=ax1)
    vis_edge(rgb_vu, indices=edges, color=edge_color, ax=ax1)

    vis_point(rgb_joint, color=color, ax=ax3)
    vis_edge(rgb_joint, indices=edges, color=edge_color, ax=ax3)

    for ax in [ax3]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(-65, -90)

    plt.savefig("output.png")
    plt.show()


def visualize_depth(dataset, keypoint_names, edges, color_map, normalize=False):
    idx = random.randint(0, len(dataset) - 1)
    logger.info("get example")
    example = dataset.get_example(idx)
    logger.info("Done get example")
    fig = plt.figure(figsize=(5, 10))
    ax2 = fig.add_subplot(211)
    ax4 = fig.add_subplot(212, projection="3d")
    color = [color_map[k] for k in keypoint_names]
    edge_color = [color_map[s, t] for s, t in edges]

    depth = example["depth"].astype(np.float32)
    depth_joint = example["depth_joint"]
    depth_camera = example["depth_camera"]
    depth_vu, depth_z = depth_camera.zyx2vu(depth_joint, return_z=True)
    z_size = example["param"]["z_size"]
    if normalize:
        depth = normalize_depth(depth, z_com=depth_z.mean(), z_size=z_size)
    depth_joint = normalize_joint_zyx(depth_joint, depth_camera, z_size)

    print(example["param"])

    vis_point(depth_vu, img=depth, color=color, ax=ax2)
    vis_edge(depth_vu, indices=edges, color=edge_color, ax=ax2)

    vis_point(depth_joint, color=color, ax=ax4)
    vis_edge(depth_joint, indices=edges, color=edge_color, ax=ax4)

    for ax in [ax4]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(-65, -90)
    plt.savefig("output.png")
    plt.show()
