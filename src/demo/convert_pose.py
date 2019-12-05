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
    from pose.hand_dataset.selector import select_dataset as select_pose_dataset
    from pose.models.selector import select_model as select_pose_model
except ImportError:
    raise Exception("Please install our `hand` package via\n `pip install -e ../../`")


def setup_pose(pose_path):
    config = configparser.ConfigParser()
    path = os.path.expanduser(os.path.join(pose_path, "pose", "config.ini"))
    logger.info("read {}".format(path))
    config.read(path, 'UTF-8')

    logger.info("setup devices")

    pose_param = select_pose_dataset(config, return_data=["hand_param"])
    model_path = os.path.expanduser(os.path.join(pose_path, "pose", "bestmodel.npz"))

    logger.info("> restore model")
    model = select_pose_model(config, pose_param)
    logger.info("> model.device = {}".format(model.device))

    logger.info("> restore models")
    chainer.serializers.load_npz(model_path, model)
    return model, pose_param


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("trained", type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    model, pose_param = setup_pose(pose_path=args.trained)
    inC = pose_param["inC"]
    inH = pose_param["inH"]
    inW = pose_param["inW"]
    pose_param["K"] = int(model.K)
    pose_param["outH"] = int(model.outH)
    pose_param["outW"] = int(model.outW)
    pose_param["inC"] = int(inC)
    pose_param["inH"] = int(inH)
    pose_param["inW"] = int(inW)

    def dump_nnoir(batch_size=1):
        # forward
        X = chainer.Variable(np.ones((batch_size, inC, inH, inW)).astype(np.float32))
        with chainer.using_config('train', False):
            model.__class__.__name__ = "PoseEstimatorBatch{}".format(batch_size)
            pose, vect = model._forward(X)

        g = nnoir_chainer.Graph(model, (X,), (pose, vect))
        result = g.to_nnoir()
        # dump result
        if batch_size == 1:
            nnoir_name = "pose.nnoir"
        else:
            nnoir_name = "pose_batch_{}.nnoir".format(batch_size)
        with open(nnoir_name, 'w') as f:
            f.buffer.write(result)

    dump_nnoir(1)
    # dump_nnoir(2)

    with open("pose_param.json", 'w') as f:
        keys = ["inC", "inH", "inW", "outH", "outW", "K", "keypoint_names", "edges"]
        pose_param = {k: pose_param[k] for k in keys}
        json.dump(pose_param, f)
