import configparser
import json
import os
import time
import logging
logger = logging.getLogger(__name__)

import traceback

import numpy as np


class HandPose():
    def __init__(self, model, param):
        self.model = model
        self.param = param
        self.inC = param["inC"]
        self.inH = param["inH"]
        self.inW = param["inW"]
        self.outH = param["outH"]
        self.outW = param["outW"]
        self.K = param["K"]
        self.keypoint_names = param["keypoint_names"]
        self.edges = param["edges"]

    def predict(self, img):
        # assume the layout of `x` is CHW or NCHW
        try:
            time_ppn = time.time()
            logger.info("> {} predict".format(self.__class__.__name__))
            if img.ndim == 3:
                img = np.expand_dims(img, axis=0)
            pose, vect = self.model(img)
            logger.info("> ppn inference {} [msec]".format(1000 * (time.time() - time_ppn)))
            time_postprocessing = time.time()
            K = self.K
            N, _, _, _ = img.shape
            outH, outW = self.outH, self.outW
            inH, inW = self.inH, self.inW
            resp = pose[:, 0 * K:1 * K, :, :]
            conf = pose[:, 1 * K:2 * K, :, :]
            xs = pose[:, 2 * K:3 * K, :, :]
            ys = pose[:, 3 * K:4 * K, :, :]
            #w = pose[:, 4 * K:5 * K, :, :]
            #h = pose[:, 5 * K:6 * K, :, :]

            vs = vect.reshape(N, -1, 3, outH, outW)
            kp_vu = []
            kp_zyx = []
            for r, c, x, y, v in zip(resp, conf, xs, ys, vs):
                keypoint_names = self.keypoint_names
                edges = self.edges
                delta = r * c
                scaleH = inH / outH
                scaleW = inW / outW
                joint2d = {}
                grid_position = {}

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
                        hslice = slice(max(0, p_h - sz), min(outH, p_h + sz + 1))
                        wslice = slice(max(0, p_w - sz), min(outW, p_w + sz + 1))
                        target = delta[i][hslice, wslice]
                        q_h, q_w = np.unravel_index(np.argmax(target), target.shape)
                        y_offset = (p_h - sz) + q_h if p_h - sz >= 0 else q_h
                        x_offset = (p_w - sz) + q_w if p_w - sz >= 0 else q_w
                        joint2d[f_q] = [
                            scaleH * (y_offset + y[i][(y_offset, x_offset)]),
                            scaleW * (x_offset + x[i][(y_offset, x_offset)])
                        ]
                        grid_position[f_q] = (y_offset, x_offset)
                vu = np.array([joint2d[k] for k in keypoint_names])

                zyx = np.zeros((len(keypoint_names), 3))
                for ei, (s, t) in enumerate(edges):
                    u_ind = grid_position[keypoint_names[s]]
                    orien = v[ei, :, u_ind[0], u_ind[1]]
                    orien = orien / np.linalg.norm(orien)
                    # elen = np.sqrt(np.sum(np.square(gt_3dj[t] - gt_3dj[s])))
                    elen = 1.5 if s == 0 else 1
                    zyx[t] = zyx[s] + orien * elen
                logger.info("> ppn_postprocessing {} [msec]".format(1000 * (time.time() - time_postprocessing)))
                kp_vu.append(vu)
                kp_zyx.append(zyx)
            return kp_vu, kp_zyx
        except Exception as e:
            logger.error("error {}".format(e))
            traceback.print_exc()
            return None, None


def setup_npz(pose_path):
    import chainer
    try:
        from pose.hand_dataset.selector import select_dataset as select_pose_dataset
        from pose.models.selector import select_model as select_pose_model
    except ImportError:
        raise ImportError("Please install our `hand` package via \n `pip install -e ../../`")
    config = configparser.ConfigParser()
    path = os.path.expanduser(os.path.join(pose_path, "pose", "config.ini"))
    logger.info("read {}".format(path))
    config.read(path, 'UTF-8')

    logger.info("setup devices")

    pose_param = select_pose_dataset(config, return_data=["hand_param"])
    model_path = os.path.expanduser(os.path.join(pose_path, "pose", "bestmodel.npz"))

    logger.info("> restore model")
    pose_model = select_pose_model(config, pose_param)
    logger.info("> model.device = {}".format(pose_model.device))

    logger.info("> restore models")
    chainer.serializers.load_npz(model_path, pose_model)

    logger.info('> cuda enable {}'.format(chainer.backends.cuda.available))
    logger.info('> ideep enable {}'.format(chainer.backends.intel64.is_ideep_available()))
    if chainer.backends.cuda.available:
        logger.info('> use GPU mode')
        pose_model.to_gpu()
    elif chainer.backends.intel64.is_ideep_available():
        logger.info('> Intel64 mode')
        pose_model.to_intel64()

    pose_param["outH"] = pose_model.outH
    pose_param["outW"] = pose_model.outW
    pose_param["K"] = pose_model.K
    chainer.global_config.train=False
    chainer.global_config.autotune=True
    chainer.global_config.use_ideep="auto"
    chainer.global_config.enable_backprop=False
    def model(x):
        x = pose_model.xp.asarray(x)
        # assume the layout of x is NCHW
        pose, vect = pose_model._forward(x)
        pose = chainer.backends.cuda.to_cpu(pose.array)
        vect = chainer.backends.cuda.to_cpu(vect.array)
        return pose, vect

    return model, pose_param


def setup_nnoir(pose_path):
    logger.info("> use nnoir")
    import chainer
    from nnoir_chainer import NNOIRFunction
    pose_model = NNOIRFunction(os.path.join(pose_path, "pose.nnoir"))
    with open(os.path.join(pose_path, "pose_param.json"), 'r') as f:
        pose_param = json.load(f)
    logger.info("> param {}".format(pose_param))

    def model(x):
        # assume the layout of x is NCHW
        with chainer.using_config("train", False):
            pose, vect = pose_model(x)
        pose, vect = pose.array, vect.array
        return pose, vect
    return model, pose_param


def select_pose(pose_path, file_type):
    if file_type == "npz":
        model, param = setup_npz(pose_path)
        return HandPose(model, param)
    elif file_type == "nnoir":
        model, param = setup_nnoir(pose_path)
        return HandPose(model, param)
    else:
        raise ValueError("Invalid file_type {}".format(file_type))
