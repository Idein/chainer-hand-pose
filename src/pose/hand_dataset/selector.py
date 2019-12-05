import os
import importlib
import logging

logger = logging.getLogger(__name__)

import numpy as np
from chainer.datasets import ConcatenatedDataset

from pose.hand_dataset.common_dataset import NUM_KEYPOINTS, EDGES, STANDARD_KEYPOINT_NAMES, COLOR_MAP
from pose.utils import parse_imsize, parse_cube, remove_whitespace

IMPLEMENTED_DATASET = ["fhad", "stb", "rhd", "msra15", "nyu", "synth", "ganerated", "multiview", "handdb", "freihand"]


class Dataset(ConcatenatedDataset):
    def __init__(self, config, param, **kwargs):
        if kwargs["mode"] == "train":
            mode = "train_set"
        if kwargs["mode"] == "val":
            mode = "val_set"
        if kwargs["mode"] == "test":
            mode = "test_set"
        dataset_type_list = remove_whitespace(config["dataset"][mode]).split(",")
        dataset_list = []
        for dataset_type in dataset_type_list:
            logger.info("dataset_type = {}".format(dataset_type))
            if not dataset_type in IMPLEMENTED_DATASET:
                raise Exception("dataset_type {} is not supported".format(dataset_type))
            x_dataset = importlib.import_module("pose.hand_dataset.{}_dataset".format(dataset_type))
            dset = x_dataset.get_dataset(os.path.expanduser(config["dataset_dir"][dataset_type]), param, **kwargs)
            logger.info("num of {} = {}".format(dataset_type, len(dset)))
            dataset_list.append(dset)
        super(Dataset, self).__init__(*dataset_list)


def select_dataset(config, return_data=["train_set", "val_set", "test_set", "hand_param"], debug=False):
    # joint dims and embedding_dim
    imsize = parse_imsize(config["training_param"]["imsize"], style="HW")
    cube = parse_cube(config["training_param"]["cube"], style="DHW")
    joint_dims = remove_whitespace(config["training_param"]["joint_dims"]).split(",")
    logger.info("> joint_dims {}".format(joint_dims))
    if len(joint_dims) == 1:
        embedding_dim = 30
        joint_dim = int(joint_dims[0])
    elif len(joint_dims) == 2:
        joint_dim = 2 + 3
        embedding_dim = 50
    else:
        raise ValueError("joint_dims must be 1 or 2 not {}".format(joint_dims))

    use_rgb = config.getboolean("dataset", "use_rgb")
    use_depth = config.getboolean("dataset", "use_depth")
    # xyflip
    enable_x_flip = config.getboolean("training_param", "enable_x_flip")
    enable_y_flip = config.getboolean("training_param", "enable_y_flip")
    # angle_range
    angle_range = config["training_param"]["angle_range"]
    angle_min, angle_max = remove_whitespace(angle_range).split(",")
    angle_min = int(angle_min)
    angle_max = int(angle_max)
    assert angle_min <= angle_max
    angle_range = range(angle_min, angle_max)
    # crop parameter
    scale_range = config["training_param"]["scale_range"]
    b, e, step = remove_whitespace(scale_range).split(",")
    scale_range = np.arange(float(b), float(e), float(step))

    shift_range = config["training_param"]["shift_range"]
    b, e, step = remove_whitespace(shift_range).split(",")
    shift_range = np.arange(float(b), float(e), float(step))

    inH, inW = imsize
    inC = 1 if use_depth else 3

    hand_param = {
        "keypoint_names": STANDARD_KEYPOINT_NAMES,
        "color_map": COLOR_MAP,
        "edges": EDGES,
        "n_joints": NUM_KEYPOINTS,
        "inC": inC,
        "inH": inH,
        "inW": inW,
        "imsize": imsize,
        "cube": cube,
        "use_rgb": use_rgb,
        "use_depth": use_depth,
        "output_dim": NUM_KEYPOINTS * joint_dim,
        "embedding_dim": embedding_dim,
        "joint_dim": joint_dim,
        "enable_x_flip": enable_x_flip,
        "enable_y_flip": enable_y_flip,
        "angle_range": angle_range,
        "oscillation": {
            "scale_range": scale_range,
            "shift_range": shift_range,
        },
    }

    returns = []
    return_data = set(return_data)
    if "train_set" in return_data:
        train_set = Dataset(config, param=hand_param, mode="train", debug=debug)
        returns.append(train_set)
    if "val_set" in return_data:
        val_set = Dataset(config, param=hand_param, mode="val", debug=debug)
        returns.append(val_set)
    if "test_set" in return_data:
        test_set = Dataset(config, param=hand_param, mode="test", debug=debug)
        returns.append(test_set)
    if "hand_param" in return_data:
        returns.append(hand_param)
    if len(returns) == 1:
        return returns[0]
    return returns
