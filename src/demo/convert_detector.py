import argparse
import configparser
import json
import logging

logger = logging.getLogger(__name__)
import os

import chainer
import numpy as np
import nnoir_chainer

try:
    import detector.utils as detector_utils
except ImportError:
    raise Exception("please install our `hand` package via\n `pip install -e ../../`")


def setup_detector(detector_path):
    logger.info("> setup config")
    config = configparser.ConfigParser()
    config.read(os.path.join(detector_path, "detector", "config.ini"))
    detector_param = detector_utils.get_config(config)
    logger.info("> setup model")
    model = detector_utils.create_ssd_model(detector_param)
    chainer.serializers.load_npz(
        os.path.join(detector_path, "detector", "bestmodel.npz"),
        model
    )
    hand_class = config["model_param"]["hand_class"].split(",")
    hand_class = [k.strip() for k in hand_class]
    return model, hand_class


def main(args):
    model, hand_class = setup_detector(args.trained)
    # will be utilized for application
    model_param = {
        "hand_class": hand_class,
        "insize": model.insize,
        "mean": model.mean,
        "nms_thresh": model.nms_thresh,
        "score_thresh": model.score_thresh,
        "grids": model.grids,
        "aspect_ratios": model.multibox.aspect_ratios,
        "steps": model.steps,
        "sizes": model.sizes,
    }
    model.__class__.__name__ = args.out_classname
    # prepare forward
    npX = np.ones((1, 3, *(model.insize, model.insize)), dtype=np.float32)
    X = chainer.Variable(npX)
    with chainer.using_config('train', False):
        mb_locs, mb_confs = model(X)

    # get computational graph for NNOIR
    g = nnoir_chainer.Graph(model, (X,), (mb_locs, mb_confs))
    result = g.to_nnoir()

    # dump result
    with open(args.out_nnoir, 'w') as f:
        f.buffer.write(result)
    with open(args.out_param, 'w') as f:
        json.dump(model_param, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("trained", type=str)
    parser.add_argument("--out-classname", type=str, default="SSD")
    parser.add_argument("--out-nnoir", type=str, default="detector.nnoir")
    parser.add_argument("--out-param", type=str, default="detector_param.json")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
