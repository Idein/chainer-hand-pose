import chainer
import numpy as np
from chainer import initializers
import chainer.functions as F
import chainer.links as L


class Convolution2d(chainer.Chain):
    """
    convert pose_estimation.network_base.convolution2d written in tensorflow.contrib.slim
    into Chainer implementation
    """

    def __init__(self, in_channels, out_channels, ksize=3, stride=1):
        super(Convolution2d, self).__init__()
        self.dtype = np.float32
        initialW = initializers.HeNormal(1 / np.sqrt(2), self.dtype)
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels,
                                        out_channels,
                                        ksize=ksize,
                                        stride=stride,
                                        pad=ksize // 2,
                                        initialW=initialW,
                                        nobias=True)
            self.bn = L.BatchNormalization(out_channels,
                                           eps=0.001, decay=0.9997)

    def __call__(self, x):
        return F.clipped_relu(self.bn(self.conv(x)), 6.0)


class Conv1x1(chainer.Chain):
    def __init__(self, in_channels, out_channels, relu=True):
        super(Conv1x1, self).__init__()
        self.relu = relu
        with self.init_scope():
            self.conv = L.Convolution2D(
                in_channels,
                out_channels,
                ksize=1,
                stride=1,
                nobias=True
            )
            self.bn = L.BatchNormalization(
                out_channels,
                eps=0.001,
                use_gamma=False
            )

    def __call__(self, x):
        h = self.bn(self.conv(x))
        if self.relu:
            return F.relu(h)
        else:
            return h


class SeparableConv(chainer.Chain):
    """
    convert pose_estimation.network_base.separable_conv written in tensorflow.contrib.slim
    into Chainer implementation
    """

    def __init__(self, in_channels, out_channels, ksize=3, stride=1, relu=True):
        super(SeparableConv, self).__init__()
        self.relu = relu
        self.ksize = ksize
        self.stride = stride
        with self.init_scope():
            self.depthwise_conv = L.DepthwiseConvolution2D(
                in_channels=in_channels,
                channel_multiplier=1,
                ksize=ksize,
                pad=ksize // 2,
                stride=stride,
                nobias=True
            )
            self.pointwise_conv = L.Convolution2D(
                in_channels,
                out_channels,
                ksize=1,
                nobias=True
            )
            self.pointwise_bn = L.BatchNormalization(
                out_channels,
                eps=0.001,
                use_gamma=False
            )

    def __call__(self, x):
        h = self.depthwise_conv(x)
        h = self.pointwise_conv(h)
        h = self.pointwise_bn(h)
        if self.relu:
            h = F.relu(h)
        return h


class ExpandedConv(chainer.Chain):

    def __init__(self, expand_ratio, in_channels, out_channels, stride):
        super(ExpandedConv, self).__init__()
        ksize = 3
        self.dtype = np.float32
        self.expand_ratio = expand_ratio
        expanded_channels = int(in_channels * expand_ratio)
        initialW = initializers.HeNormal(1 / np.sqrt(2), self.dtype)
        with self.init_scope():
            if expand_ratio != 1:
                self.expand_conv = L.Convolution2D(in_channels,
                                                   expanded_channels,
                                                   ksize=1,
                                                   initialW=initialW,
                                                   nobias=True)
                self.expand_bn = L.BatchNormalization(expanded_channels,
                                                      eps=0.001, decay=0.997)

            self.depthwise_conv = L.DepthwiseConvolution2D(expanded_channels,
                                                           channel_multiplier=1,
                                                           ksize=ksize,
                                                           stride=stride,
                                                           pad=ksize // 2,
                                                           initialW=initialW,
                                                           nobias=True)
            self.depthwise_bn = L.BatchNormalization(expanded_channels,
                                                     eps=0.001, decay=0.9997)
            self.project_conv = L.Convolution2D(expanded_channels,
                                                out_channels,
                                                ksize=1,
                                                initialW=initialW,
                                                nobias=True)
            self.project_bn = L.BatchNormalization(out_channels,
                                                   eps=0.001, decay=0.9997)

    def __call__(self, x):
        h = x
        if self.expand_ratio != 1:
            h = F.clipped_relu(self.expand_bn(self.expand_conv(h)), 6.0)
        h = F.clipped_relu(self.depthwise_bn(self.depthwise_conv(h)), 6.0)
        h = self.project_bn(self.project_conv(h))
        if h.shape == x.shape:
            return h + x
        else:
            return h
