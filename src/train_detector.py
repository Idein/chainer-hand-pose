# -*- coding: utf-8 -*-

try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:
    pass

import cv2
cv2.setNumThreads(0)

import argparse
import copy
import os
import configparser

import chainer
import chainercv
from chainer.datasets import TransformDataset
from chainer.optimizer import WeightDecay
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import triggers
import numpy as np
import shutil

from detector.hand_dataset.selector import select_dataset

from detector.evaluation.detection_coco_evaluator import DetectionCOCOEvaluator
from detector.models.gradient_scaling import GradientScaling
from detector.models.multibox_loss import multibox_loss
from detector.models.multibox_focal_loss import multibox_focal_loss

from detector import utils
from detector.minmax_value_trigger import MaxValueTrigger

from logging import getLogger

logger = getLogger(__name__)


class MultiboxTrainChain(chainer.Chain):
    def __init__(self, model, alpha=1, beta=1, k=3):
        super(MultiboxTrainChain, self).__init__()
        with self.init_scope():
            self.model = model
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def __call__(self, imgs, gt_mb_locs, gt_mb_labels):
        mb_locs, mb_confs = self.model(imgs)
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, self.k)
        loss = loc_loss * self.alpha + conf_loss * self.beta

        chainer.reporter.report(
            {"loss": loss, "loss/loc": loc_loss, "loss/conf": conf_loss},
            self)

        return loss


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
            # 2. Random expansion
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
                    y_offset=param["y_offset"],
                    x_offset=param["x_offset"],
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
                y_slice=param["y_slice"],
                x_slice=param["x_slice"],
                allow_outside_center=False,
                return_param=True,
            )
            label = label[param["index"]]

            # 4. Resizing with random interpolatation
            _, H, W = img.shape
            img = chainercv.links.model.ssd.resize_with_random_interpolation(img, (self.size, self.size))
            bbox = chainercv.transforms.resize_bbox(bbox, (H, W), (self.size, self.size))

            # Preparation for SSD network
            img -= self.mean
            mb_loc, mb_label = self.coder.encode(bbox, label, iou_thresh=0.35)
            return img, mb_loc, mb_label
        else:
            return img, bbox, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config.ini")
    parser.add_argument("--resume")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path, "UTF-8")
    train_param = utils.get_config(config)

    chainer.global_config.autotune = True
    chainer.cuda.set_max_workspace_size(11388608)
    chainer.config.cudnn_fast_batch_normalization = True
    logger.info("> set up devices")
    if chainer.backends.cuda.available:
        devices = utils.setup_devices(train_param["gpus"])
    else:
        # cpu run
        devices = {"main": -1}
    logger.info("> set devices {}".format(devices))
    utils.set_random_seed(devices, train_param["seed"])

    # get dataset
    logger.info("> get dataset")
    train, test = select_dataset(config, return_data=["train_set", "val_set"])
    logger.info("> size of train {}".format(len(train)))
    logger.info("> size of test {}".format(len(test)))
    # create result dir and copy file
    result=config["output_path"]["result_dir"]
    logger.info("> store file to result dir {}".format(result))
    utils.create_result_dir(result)
    destination=os.path.join(result,"detector")
    logger.info("> store config.ini to {}".format(os.path.join(destination, "config.ini")))
    if not os.path.exists(destination):
        os.makedirs(destination)
    shutil.copy(args.config_path, os.path.join(destination, "config.ini"))
    # load model
    logger.info("> load model")
    model = utils.create_ssd_model(train_param)

    model.use_preset("evaluate")
    train_chain = MultiboxTrainChain(model, beta=4)

    logger.info("> transform dataset")

    train = TransformDataset(
        train,
        Transform(model.coder, model.insize, model.mean, train=True))
    train_iter = chainer.iterators.MultiprocessIterator(
        train, train_param["batchsize"],
        n_processes=train_param["num_process"]
    )

    test = TransformDataset(
        test,
        Transform(model.coder, model.insize, model.mean, train=False)
    )
    test_iter = chainer.iterators.MultiprocessIterator(
        test, train_param["batchsize"],
        repeat=False, shuffle=False,
        n_processes=4
    )

    # initial lr is set to 1e-3 by ExponentialShift
    logger.info("> set up optimizer")
    optimizer = chainer.optimizers.MomentumSGD(lr=train_param["learning_rate"])
    # optimizer = chainer.optimizers.RMSprop(lr=train_param["learning_rate"])
    optimizer.setup(train_chain)
    for param in train_chain.params():
        if param.name == "b":
            param.update_rule.add_hook(GradientScaling(2))
        else:
            param.update_rule.add_hook(WeightDecay(0.0005))

    updater = training.ParallelUpdater(train_iter, optimizer, devices=devices)
    trainer = training.Trainer(
        updater, (train_param["train_iter"], "iteration"),
        destination,
    )
    trainer.extend(
        extensions.ExponentialShift("lr", 0.1, init=train_param["learning_rate"]),
        trigger=triggers.ManualScheduleTrigger(train_param["schedule"], "iteration")
    )

    # set current device to devices["main"]
    # with chainer.cuda.Device(devices["main"]):
    eval_interval = 500, "iteration"
    logger.info("setup evaluator {}".format(train_param["hand_class"]))
    trainer.extend(
        DetectionCOCOEvaluator(
            test_iter,
            model,
            device=devices["main"],
            label_names=train_param["hand_class"],
        ),
        trigger=eval_interval,
    )

    log_interval = 100, "iteration"
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ["epoch", "iteration", "lr",
         "main/loss", "main/loss/loc", "main/loss/conf",
         "validation/main/map",
         ]
    ),
        trigger=log_interval
    )
    trainer.extend(extensions.ProgressBar(update_interval=100))

    trainer.extend(extensions.snapshot(filename="best_snapshot"),
                   trigger=MaxValueTrigger("validation/main/map", trigger=eval_interval))
    trainer.extend(extensions.snapshot_object(model, filename="bestmodel.npz"),
                   trigger=MaxValueTrigger("validation/main/map", trigger=eval_interval))

    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(
            ["main/loss", "validation/main/loss"], x_key="iteration", file_name="loss.png"))
        trainer.extend(extensions.PlotReport(
            ["main/accuracy/map", "validation/main/map"], x_key="iteration",
            file_name="accuracy.png"))

    if args.resume:
        serializers.load_npz(args.resume, trainer)

    logger.info("> run trainer")
    trainer.run()


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
