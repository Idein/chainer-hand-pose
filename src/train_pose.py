import argparse
import configparser
import logging

logger = logging.getLogger(__name__)

import os

import cv2

cv2.setNumThreads(0)

import matplotlib

matplotlib.use("Agg")

import chainer
from chainer.training import extensions
from chainer import training
from chainer.datasets import TransformDataset
import shutil

from pose.hand_dataset.selector import select_dataset
from pose.models.selector import select_model
from pose.utils import save_files
from pose.utils import setup_devices, set_random_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="config.ini")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    config = configparser.ConfigParser()

    logger.info("read {}".format(args.config_path))
    config.read(args.config_path, "UTF-8")
    logger.info("setup devices")
    if chainer.backends.cuda.available:
        devices = setup_devices(config["training_param"]["gpus"])
    else:
        # cpu run
        devices = {"main": -1}
    seed = config.getint("training_param", "seed")
    logger.info("set random seed {}".format(seed))
    set_random_seed(devices, seed)

    result = os.path.expanduser(config["result"]["dir"])
    destination = os.path.join(result, "pose")
    logger.info("> copy code to {}".format(os.path.join(result, "src")))
    save_files(result)
    logger.info("> copy config file to {}".format(destination))
    if not os.path.exists(destination):
        os.makedirs(destination)
    shutil.copy(args.config_path, os.path.join(destination, "config.ini"))

    logger.info("{} chainer debug".format("enable" if args.debug else "disable"))
    chainer.set_debug(args.debug)
    chainer.global_config.autotune = True
    chainer.cuda.set_max_workspace_size(11388608)
    chainer.config.cudnn_fast_batch_normalization = True

    logger.info("> get dataset")
    train_set, val_set, hand_param = select_dataset(config, return_data=["train_set", "val_set", "hand_param"])
    model = select_model(config, hand_param)

    logger.info("> transform dataset")
    train_set = TransformDataset(train_set, model.encode)
    val_set = TransformDataset(val_set, model.encode)
    logger.info("> size of train_set is {}".format(len(train_set)))
    logger.info("> size of val_set is {}".format(len(val_set)))
    logger.info("> create iterators")
    batch_size = config.getint("training_param", "batch_size")
    n_processes = config.getint("training_param", "n_processes")

    train_iter = chainer.iterators.MultiprocessIterator(
        train_set, batch_size,
        n_processes=n_processes
    )
    test_iter = chainer.iterators.MultiprocessIterator(
        val_set, batch_size,
        repeat=False, shuffle=False,
        n_processes=n_processes,
    )

    logger.info("> setup optimizer")
    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

    logger.info("> setup parallel updater devices={}".format(devices))
    updater = training.updaters.ParallelUpdater(train_iter, optimizer, devices=devices)

    logger.info("> setup trainer")
    trainer = training.Trainer(
        updater,
        (config.getint("training_param", "train_iter"), "iteration"),
        destination,
    )

    logger.info("> setup extensions")
    trainer.extend(
        extensions.LinearShift("lr",
                               value_range=(config.getfloat("training_param", "learning_rate"), 0),
                               time_range=(0, config.getint("training_param", "train_iter"))
                               ),
        trigger=(1, "iteration")
    )

    trainer.extend(extensions.Evaluator(test_iter, model, device=devices["main"]))
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport([
            "main/loss", "validation/main/loss",
        ], "epoch", file_name="loss.png"))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr())
    trainer.extend(extensions.PrintReport([
        "epoch", "elapsed_time", "lr",
        "main/loss", "validation/main/loss",
        "main/loss_resp", "validation/main/loss_resp",
        "main/loss_iou", "validation/main/loss_iou",
        "main/loss_coor", "validation/main/loss_coor",
        "main/loss_size", "validation/main/loss_size",
        "main/loss_limb", "validation/main/loss_limb",
        "main/loss_vect_cos", "validation/main/loss_vect_cos",
        "main/loss_vect_norm", "validation/main/loss_vect_cos",
        "main/loss_vect_square", "validation/main/loss_vect_square",
    ]))
    trainer.extend(extensions.ProgressBar())

    trainer.extend(extensions.snapshot(filename="best_snapshot"),
                   trigger=training.triggers.MinValueTrigger("validation/main/loss"))
    trainer.extend(extensions.snapshot_object(model, filename="bestmodel.npz"),
                   trigger=training.triggers.MinValueTrigger("validation/main/loss"))

    logger.info("> start training")
    trainer.run()


if __name__ == "__main__":
    import logging

    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    main()
