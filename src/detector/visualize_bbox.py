import configparser
import copy
import logging

logger = logging.getLogger(__name__)

import random

from chainer.datasets import TransformDataset
from chainercv.visualizations import vis_bbox
import chainercv
import numpy as np

from detector.hand_dataset.image_utils import COLOR_MAP
from detector.hand_dataset.selector import select_dataset
from detector.hand_dataset.bbox_dataset import create_converter


class Transform(object):
    def __init__(self, coder, size, mean, train):
        # to send cpu, make a copy
        self.coder = copy.copy(coder)
        self.coder.to_cpu()

        self.size = size
        self.mean = mean
        self.train = train

    def __call__(self, in_data):
        img, bbox, label = in_data
        if self.train:
            # There are five data augmentation steps
            # 1. Color augmentation
            # 2. Random expansion <- disabled
            # 3. Random cropping
            # 4. Resizing with random interpolation

            # 1. Color augmentation
            img = chainercv.links.model.ssd.random_distort(img)

            # 2. Random expansion
            if np.random.randint(2):
                img, param = chainercv.transforms.random_expand(
                    img,
                    fill=self.mean,
                    return_param=True,
                    max_ratio=1.5,
                )
                bbox = chainercv.transforms.translate_bbox(
                    bbox,
                    y_offset=param['y_offset'],
                    x_offset=param['x_offset'],
                )
            # 3. Random cropping
            img, param = chainercv.links.model.ssd.random_crop_with_bbox_constraints(
                img,
                bbox,
                min_scale=0.5,
                max_aspect_ratio=1.25,
                return_param=True,
            )
            bbox, param = chainercv.transforms.crop_bbox(
                bbox,
                y_slice=param['y_slice'],
                x_slice=param['x_slice'],
                allow_outside_center=False,
                return_param=True,
            )
            label = label[param['index']]

            # 4. Resizing with random interpolatation
            _, H, W = img.shape
            img = chainercv.links.model.ssd.resize_with_random_interpolation(img, (self.size, self.size))
            bbox = chainercv.transforms.resize_bbox(bbox, (H, W), (self.size, self.size))

            # Preparation for SSD network
            img -= self.mean
            mb_loc, mb_label = self.coder.encode(bbox, label, iou_thresh=0.35)
            # return img, mb_loc, mb_label
            return img, bbox, label
        else:
            return img, bbox, label


def create_ssd_model():
    import os
    from importlib import import_module
    from detector.models.ssd_v2 import SSD
    model_path = '.'.join(os.path.split("models/face_ssd_mobilenet_v2"))
    model = import_module(model_path)
    feature_extractor = getattr(model, "MobileNetV2")(1.0)
    ssd_extractor = getattr(model, "MobileNetV2LiteExtractor300")(feature_extractor, 1.0)

    model = SSD(
        ssd_extractor, input_size=int(256 * 1.0),
        n_fg_class=2,
        num_layers=6,
        smin=0.025, smax=0.8,
        # aspect_ratios=((2,), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3)),
        aspect_ratios=((1,), (1,), (1,), (1,), (1,), (1,)),
        variance=(0.1, 0.2), mean=0
    )
    return model


def visualize_dataset(config):
    from matplotlib import pyplot as plt
    dataset = select_dataset(config, return_data=["train_set"])
    hand_class = config.get('model_param', 'hand_class').split(",")
    hand_class = [k.strip() for k in hand_class]
    class_converter, flip_converter = create_converter(hand_class)
    logger.info("hand_class = {}".format(hand_class))
    logger.info("done get dataset")

    idx = random.randint(0, len(dataset) - 1)
    logger.info("get example")
    rgb, rgb_bbox, rgb_class = dataset.get_example(idx)
    logger.info("Done get example")
    fig = plt.figure(figsize=(5, 10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    label = rgb_class
    class_converter = {v: k for k, v in class_converter.items()}
    color = [COLOR_MAP[class_converter[c]] for c in label]
    print(label)
    vis_bbox(
        rgb,
        rgb_bbox,
        instance_colors=color,
        label=label,
        label_names=hand_class,
        ax=ax1,
    )

    model = create_ssd_model()
    transform_dataset = TransformDataset(
        dataset,
        Transform(model.coder, model.insize, model.mean, train=True)
    )

    img, mb_loc, mb_label = transform_dataset.get_example(idx)
    mb_color = [COLOR_MAP[class_converter[c]] for c in mb_label]
    vis_bbox(
        img,
        mb_loc,
        instance_colors=mb_color,
        label=mb_label,
        label_names=hand_class,
        ax=ax2,
    )
    plt.savefig("vis.png")
    plt.show()


if __name__ == "__main__":
    dataset_type = "stb"
    import cv2

    config = configparser.ConfigParser()
    config.read("config.ini")
    config["dataset"]["train_set"] = dataset_type
    logging.basicConfig(level=logging.INFO)
    cv2.setNumThreads(0)
    visualize_dataset(config)
