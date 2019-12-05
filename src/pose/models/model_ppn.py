import math
import os

import chainer
from chainer.backends.cuda import get_array_module
from chainer import reporter
import chainer.functions as F
import chainer.links as L
from chainer import initializers
import numpy as np

from pose.hand_dataset.image_utils import normalize_rgb

EPSILON = 1e-6


def calc_unit_vector(vector):
    return vector / (np.linalg.norm(vector) + EPSILON)


def variable_norm(variable, axis=2):
    return F.sqrt(EPSILON + F.sum(variable * variable, axis=2))


def area(bbox):
    _, _, w, h = bbox
    return w * h


def intersection(bbox0, bbox1):
    x0, y0, w0, h0 = bbox0
    x1, y1, w1, h1 = bbox1

    w = F.relu(F.minimum(x0 + w0 / 2, x1 + w1 / 2) - F.maximum(x0 - w0 / 2, x1 - w1 / 2))
    h = F.relu(F.minimum(y0 + h0 / 2, y1 + h1 / 2) - F.maximum(y0 - h0 / 2, y1 - h1 / 2))

    return w * h


def iou(bbox0, bbox1):
    area0 = area(bbox0)
    area1 = area(bbox1)
    intersect = intersection(bbox0, bbox1)

    return intersect / (area0 + area1 - intersect + EPSILON)


def get_network(model_name, **kwargs):
    if model_name == 'mv2':
        from pose.models.network_mobilenetv2 import MobilenetV2
        return MobilenetV2(**kwargs)
    elif model_name == 'resnet18':
        from pose.models.network_resnet import ResNet, AdditionalLayer
        resnet = chainer.Sequential(ResNet(n_layers=18), AdditionalLayer(ch=512))
        return resnet
    elif model_name == 'resnet34':
        from pose.models.network_resnet import ResNet, AdditionalLayer
        resnet = chainer.Sequential(ResNet(n_layers=34), AdditionalLayer(ch=512))
        return resnet
    else:
        raise Exception('Invalid model name')


class PoseProposalNet(chainer.link.Chain):
    def __init__(self, hand_param, model_param):
        super(PoseProposalNet, self).__init__()
        inC, inH, inW = hand_param["inC"], hand_param["inH"], hand_param["inW"]

        self.dtype = np.float32
        self.inC = inC
        self.inH = inH
        self.inW = inW
        self.insize = (inH, inW)

        self.n_joints = hand_param["n_joints"]
        self.K = self.n_joints
        self.edges = hand_param["edges"]
        self.E = len(self.edges)

        self.lambda_resp = 1.0
        self.lambda_iou = 1.0
        self.lambda_coor = 5.0
        self.lambda_size = 0.5

        self.lambda_vect_cos = 2.
        self.lambda_vect_norm = 0.1
        self.lambda_vect_square = 0.1

        with self.init_scope():
            self.feature_layer = get_network(**model_param)
            ksize = 3
            self.lastconv = L.Convolution2D(None,
                                            6 * self.K,
                                            ksize=ksize, stride=1, pad=ksize // 2,
                                            initialW=initializers.HeNormal(1 / np.sqrt(2), self.dtype))
            self.vectconv = L.Convolution2D(None,
                                            3 * len(self.edges),
                                            ksize=ksize, stride=1, pad=ksize // 2,
                                            initialW=initializers.HeNormal(1 / np.sqrt(2), self.dtype))

        if os.path.exists('ppn_2d_pre.npz'):
            chainer.serializers.load_npz('ppn_2d_pre.npz', self)
        self.outsize = self.get_outsize()
        outH, outW = self.outsize
        self.gridsize = (int(inW / outW), int(inH / outH))
        self.outH, self.outW = outH, outW
        self.parts_scale = np.array([0.2, 0.2])

    def get_outsize(self):
        inp = np.zeros((2, self.inC, self.inH, self.inW), dtype=np.float32)
        pose, vect = self._forward(inp)
        _, _, h, w = pose.shape
        return h, w

    def restore_xy(self, x, y):
        xp = get_array_module(x)
        gridW, gridH = self.gridsize
        outH, outW = self.outsize
        X, Y = xp.meshgrid(xp.arange(outW, dtype=xp.float32), xp.arange(outH, dtype=xp.float32))
        return (x + X) * gridW, (y + Y) * gridH

    def restore_size(self, w, h):
        inH, inW = self.insize
        return inW * w, inH * h

    def encode(self, example):
        rgb = example['rgb']
        pts_3d = example["rgb_joint"]
        camera = example["rgb_camera"]
        pts_2d = camera.zyx2vu(pts_3d)

        K = self.K
        is_labeled = K * [True]
        inH, inW = self.insize
        outH, outW = self.outsize
        gridW, gridH = self.gridsize

        delta = np.zeros((K, outH, outW), dtype=np.float32)
        tx = np.zeros((K, outH, outW), dtype=np.float32)
        ty = np.zeros((K, outH, outW), dtype=np.float32)
        tw = np.zeros((K, outH, outW), dtype=np.float32)
        th = np.zeros((K, outH, outW), dtype=np.float32)

        # define orientations
        tv = np.zeros(
            (3 * self.E, outH, outW),  # exclude instance count
            dtype=np.float32
        )

        y = np.min(pts_2d[:, 0])
        x = np.min(pts_2d[:, 1])
        h = np.max(pts_2d[:, 0]) - y
        w = np.max(pts_2d[:, 1]) - x

        # Set delta^i_k
        sizeH, sizeW = self.parts_scale * math.sqrt(w * w + h * h)
        cy = y + h / 2
        cx = x + w / 2
        for k, yx in enumerate(pts_2d):
            cy = yx[0] / gridH
            cx = yx[1] / gridW
            ix, iy = int(cx), int(cy)
            if 0 <= iy < outH and 0 <= ix < outW:
                delta[k, iy, ix] = 1
                tx[k, iy, ix] = cx - ix
                ty[k, iy, ix] = cy - iy
                tw[k, iy, ix] = sizeW / inW
                th[k, iy, ix] = sizeH / inH
        for ei, (s, t) in enumerate(self.edges):
            src_yx = pts_2d[s]
            iyx = (int(src_yx[0] / gridH), int(src_yx[1] / gridW))
            if iyx[0] < 0 or iyx[1] < 0 or iyx[0] >= outH or iyx[1] >= outW:
                continue
            # define tv
            unit_vec = calc_unit_vector(pts_3d[t] - pts_3d[s])
            tv[3 * ei, iyx[0], iyx[1]] = unit_vec[0]
            tv[3 * ei + 1, iyx[0], iyx[1]] = unit_vec[1]
            tv[3 * ei + 2, iyx[0], iyx[1]] = unit_vec[2]
        # preprocess image
        rgb = normalize_rgb(rgb)

        return rgb, delta, tx, ty, tw, th, tv

    def _forward(self, x):
        h = self.feature_layer(x)
        pose = F.sigmoid(self.lastconv(h))
        vect = F.tanh(self.vectconv(h))
        return pose, vect

    def forward(self, image, delta, tx, ty, tw, th, tv):
        K = self.K
        B, _, _, _ = image.shape
        outH, outW = self.outsize

        pose, vect = self._forward(image)
        resp = pose[:, 0 * K:1 * K, :, :]
        conf = pose[:, 1 * K:2 * K, :, :]
        x = pose[:, 2 * K:3 * K, :, :]
        y = pose[:, 3 * K:4 * K, :, :]
        w = pose[:, 4 * K:5 * K, :, :]
        h = pose[:, 5 * K:6 * K, :, :]

        (rx, ry), (rw, rh) = self.restore_xy(x, y), self.restore_size(w, h)
        (rtx, rty), (rtw, rth) = self.restore_xy(tx, ty), self.restore_size(tw, th)
        ious = iou((rx, ry, rw, rh), (rtx, rty, rtw, rth))

        xp = get_array_module(delta)
        tv = tv.reshape(B, -1, 3, outH, outW)
        v = vect.reshape(B, -1, 3, outH, outW)
        dots = F.sum(v * tv, axis=2)
        cos = dots / (variable_norm(v, axis=2) * xp.linalg.norm(tv, axis=2) + EPSILON)
        weight_vect = xp.zeros((B, len(self.edges), outH, outW))
        weight_vect[xp.sum(tv * tv, axis=2) != 0] = 1
        dest_cos = weight_vect  # must copy
        vnorm = variable_norm(v, axis=2)
        tvnorm = xp.linalg.norm(tv, axis=2)
        weight_norm = xp.zeros((B, len(self.edges), outH, outW))
        weight_norm[xp.sum(tv * tv, axis=2) > 0.1] = 1

        weight_square = xp.zeros(tv.shape)
        weight_square[tv != 0] = 1

        loss_resp = F.sum(F.square(resp - delta), axis=tuple(range(1, resp.ndim)))
        loss_iou = F.sum(delta * F.square(conf - ious), axis=tuple(range(1, conf.ndim)))
        loss_coor = F.sum(delta * (F.square(x - tx) + F.square(y - ty)), axis=tuple(range(1, x.ndim)))
        loss_size = F.sum(delta * (F.square(F.sqrt(w + EPSILON) - F.sqrt(tw + EPSILON)) +
                                   F.square(F.sqrt(h + EPSILON) - F.sqrt(th + EPSILON))),
                          axis=tuple(range(1, w.ndim)))
        loss_vect_cos = F.sum(F.square(weight_vect * (cos - dest_cos)), axis=tuple(range(1, weight_vect.ndim)))
        loss_vect_norm = F.sum(F.square(weight_norm * (vnorm - tvnorm)), axis=tuple(range(1, weight_norm.ndim)))
        loss_vect_square = F.sum(F.square(weight_square * (v - tv)), axis=tuple(range(1, weight_square.ndim)))

        loss_resp = F.mean(loss_resp)
        loss_iou = F.mean(loss_iou)
        loss_coor = F.mean(loss_coor)
        loss_size = F.mean(loss_size)
        loss_vect_cos = F.mean(loss_vect_cos)
        loss_vect_norm = F.mean(loss_vect_norm)
        loss_vect_square = F.mean(loss_vect_square)
        loss = self.lambda_resp * loss_resp + \
            self.lambda_iou * loss_iou + \
            self.lambda_coor * loss_coor + \
            self.lambda_size * loss_size + \
            self.lambda_vect_cos * loss_vect_cos + \
            self.lambda_vect_norm * loss_vect_norm + \
            self.lambda_vect_square * loss_vect_square

        reporter.report({
            'loss': loss,
            'loss_resp': loss_resp,
            'loss_iou': loss_iou,
            'loss_coor': loss_coor,
            'loss_size': loss_size,
            'loss_vect_cos': loss_vect_cos,
            'loss_vect_norm': loss_vect_norm,
            'loss_vect_square': loss_vect_square,
        }, self)

        return loss

    def predict(self, image):
        K = self.K
        B, _, _, _ = image.shape
        outH, outW = self.outsize

        with chainer.using_config('train', False):
            pose, vect = self._forward(image)

        resp = pose[:, 0 * K:1 * K, :, :]
        conf = pose[:, 1 * K:2 * K, :, :]
        x = pose[:, 2 * K:3 * K, :, :]
        y = pose[:, 3 * K:4 * K, :, :]
        w = pose[:, 4 * K:5 * K, :, :]
        h = pose[:, 5 * K:6 * K, :, :]

        v = vect.reshape(B, -1, 3, outH, outW)
        return resp, conf, x, y, w, h, v
