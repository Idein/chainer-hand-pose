import configparser
import json
import os
import time
import logging
logger = logging.getLogger(__name__)

import traceback

import numpy as np

from transforms import resize_bbox
from multibox_coder import MultiBoxCoder


class HandDetector():
    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.mean = self.param["mean"]
        self.insize = self.param["insize"]
        self.nms_thresh = self.param["nms_thresh"]
        self.score_thresh = self.param["score_thresh"]

        self.decoder = MultiBoxCoder(
            param["grids"],
            param["aspect_ratios"],
            param["steps"],
            param["sizes"],
        )

    def predict(self, x, sizeH, sizeW):
        try:
            detector_time = time.time()
            logger.info("> {} predict".format(self.__class__.__name__))
            if x.ndim == 3:
                x = np.expand_dims(x, axis=0)
            mb_locs, mb_confs = self.model(x)
            mb_loc = mb_locs[0]
            mb_conf = mb_confs[0]
            size = (sizeH, sizeW)
            bbox, label, score = self.decoder.decode(
                mb_loc,
                mb_conf,
                self.nms_thresh,
                self.score_thresh,
            )
            bbox = resize_bbox(
                bbox,
                (self.insize, self.insize),
                size,
            )
            logger.info("> detector_inference {} [msec]".format(1000 * (time.time() - detector_time)))
            return bbox, label, score
        except Exception as e:
            logger.error("error {}".format(e))
            traceback.print_exc()
            return None, None, None


def setup_npz(detector_path):
    import chainer
    try:
        import detector.utils as detector_utils
    except ImportError:
        raise ImportError("Please install our `hand` package via \n `pip install -e ../../`")
    logger.info("> use npz")
    logger.info("> setup config")
    config = configparser.ConfigParser()
    logger.info("> load config {}".format(os.path.join(detector_path, "detector", "config.ini")))
    config.read(os.path.join(detector_path, "detector", "config.ini"))
    detector_param = detector_utils.get_config(config)
    logger.info("> setup model")
    detector = detector_utils.create_ssd_model(detector_param)
    chainer.serializers.load_npz(
        os.path.join(detector_path, "detector", "bestmodel.npz"),
        detector
    )
    hand_class = config["model_param"]["hand_class"].split(",")
    hand_class = [k.strip() for k in hand_class]

    detector_param = {
        "hand_class": hand_class,
        "insize": detector.insize,
        "nms_thresh": detector.nms_thresh,
        "score_thresh": detector.score_thresh,
        "mean": detector.mean,
        "grids": detector.grids,
        "aspect_ratios": detector.multibox.aspect_ratios,
        "steps": detector.steps,
        "sizes": detector.sizes,
    }
    logger.info("> param {}".format(detector_param))

    logger.info('> cuda enable {}'.format(chainer.backends.cuda.available))
    logger.info('> ideep enable {}'.format(chainer.backends.intel64.is_ideep_available()))
    if chainer.backends.cuda.available:
        logger.info('> use GPU mode')
        detector.to_gpu()
    elif chainer.backends.intel64.is_ideep_available():
        logger.info('> Intel64 mode')
        detector.to_intel64()
    chainer.global_config.train=False
    chainer.global_config.autotune=True
    chainer.global_config.use_ideep="auto"
    chainer.global_config.enable_backprop=False
    def model(x):
        x = detector.xp.asarray(x)
        # assume the layout of x is NCHW
        mb_locs, mb_confs = detector(x)
        mb_locs = chainer.backends.cuda.to_cpu(mb_locs.array)
        mb_confs = chainer.backends.cuda.to_cpu(mb_confs.array)
        return mb_locs, mb_confs
    return model, detector_param


def setup_nnoir(detector_path):
    logger.info("> use nnoir")
    import chainer
    from nnoir_chainer import NNOIRFunction
    detector = NNOIRFunction(os.path.join(detector_path, "detector.nnoir"))
    with open(os.path.join(detector_path, "detector_param.json"), 'r') as f:
        detector_param = json.load(f)
    logger.info("> param {}".format(detector_param))

    def model(x):
        # assume the layout of x is NCHW
        with chainer.using_config("train", False):
            mb_locs, mb_confs = detector(x)
        mb_locs, mb_confs = mb_locs.array, mb_confs.array
        return mb_locs, mb_confs
    return model, detector_param


def select_detector(detector_path, file_type):
    if file_type == "npz":
        model, param = setup_npz(detector_path)
        return HandDetector(model, param)
    elif file_type == "nnoir":
        model, param = setup_nnoir(detector_path)
        return HandDetector(model, param)
    else:
        raise ValueError("Invalid file_type {}".format(file_type))
