import os
import importlib
import logging

logger = logging.getLogger(__name__)

from chainer.datasets import ConcatenatedDataset

from detector.hand_dataset.utils import remove_whitespace

IMPLEMENTED_DATASET = ["fhad", "stb", "rhd", "synth", "multiview", "handdb", "freihand"]


class Dataset(ConcatenatedDataset):
    def __init__(self, config, param, **kwargs):
        mode = ""
        if kwargs["mode"] == "train":
            mode = "train_set"
        elif kwargs["mode"] == "val":
            mode = "val_set"
        elif kwargs["mode"] == "test":
            mode = "test_set"
        else:
            Exception("mode should be train_set, val_set or test_set")
        dataset_type_list = remove_whitespace(config["dataset"][mode]).split(",")
        dataset_list = []
        for dataset_type in dataset_type_list:
            logger.info("dataset_type = {}".format(dataset_type))
            if not dataset_type in IMPLEMENTED_DATASET:
                raise Exception("dataset_type {} is not supported".format(dataset_type))
            x_dataset = importlib.import_module("detector.hand_dataset.{}_dataset".format(dataset_type))
            dset = x_dataset.get_dataset(os.path.expanduser(config["dataset_dir"][dataset_type]), param, **kwargs)
            logger.info("num of {} = {}".format(dataset_type, len(dset)))
            dataset_list.append(dset)
        super(Dataset, self).__init__(*dataset_list)


def select_dataset(config, return_data=("train_set", "val_set", "test_set", "hand_param"), debug=False):
    # xyflip
    enable_x_flip = config.getboolean("dataset", "enable_x_flip")
    enable_y_flip = config.getboolean("dataset", "enable_y_flip")
    # angle_range
    angle_range = config["dataset"]["angle_range"]
    angle_min, angle_max = remove_whitespace(angle_range).split(",")
    angle_min = int(angle_min)
    angle_max = int(angle_max)
    assert angle_min <= angle_max
    angle_range = range(angle_min, angle_max)

    hand_class = config.get('model_param', 'hand_class').split(",")
    hand_class = [k.strip() for k in hand_class]
    logger.info("hand_class = {}".format(hand_class))

    hand_param = {
        "enable_x_flip": enable_x_flip,
        "enable_y_flip": enable_y_flip,
        "angle_range": angle_range,
        "hand_class": hand_class,
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
