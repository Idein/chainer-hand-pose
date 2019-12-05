from __future__ import division

import numpy as np
import warnings

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from chainercv.links.model.ssd import Multibox
from chainercv.links.model.ssd import Normalize
from chainercv.links.model.ssd import SSD
from chainercv.utils import download_model
import math

try:
    import cv2  # NOQA
    _available = True
except ImportError:
    _available = False


# RGB, (C, 1, 1) format
_imagenet_mean = np.array((123, 117, 104)).reshape((-1, 1, 1))


class Conv(chainer.link.Chain):
    def __init__(self, in_channels, out_channels, ksize=None, stride=1):
        super(Conv, self).__init__()
        self.ksize = ksize
        self.stride = stride
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels, out_channels, ksize, stride, ksize//2, nobias=True)
            self.bn = L.BatchNormalization(out_channels, eps=0.001, decay = 0.9997)

    def __call__(self, x):
        #return F.relu(self.bn(self.conv(x)))
        return F.clipped_relu(self.bn(self.conv(x)), 6.0)


class FeatureMap(chainer.link.Chain):
    def __init__(self, in_channels, intermediate_channels, out_channels):
        super(FeatureMap, self).__init__()
        self.intermediate_channels = intermediate_channels
        self.out_channels = out_channels
        with self.init_scope():
            self.conv1 = Conv(in_channels, intermediate_channels, 1)
            self.conv2 = Conv(intermediate_channels, out_channels, 3, 2)

    def __call__(self, x):
        return F.clipped_relu(self.conv2(self.conv1(x)), 6.0)


class FeatureMap_Lite(chainer.link.Chain):
    def __init__(self, in_channels, intermediate_channels, out_channels):
        super(FeatureMap_Lite, self).__init__()
        self.intermediate_channels = intermediate_channels
        self.out_channels = out_channels
        with self.init_scope():
            self.conv1 = Conv(in_channels, intermediate_channels, 1)
            self.depthwise = DepthWise_Conv(intermediate_channels, intermediate_channels, 3, 2)
            self.pointwise = PointWise_Conv(intermediate_channels, out_channels, 1, 1)

    def __call__(self, x):
        #return F.clipped_relu(self.pointwise(self.depthwise(x)), 6.0)
        return F.clipped_relu(self.pointwise(self.depthwise(self.conv1(x))), 6.0)



class MN_Bottleneck(chainer.link.Chain):
    def __init__(self, t_expansion, in_channels, out_channels, ksize, stride):
        super(MN_Bottleneck, self).__init__()
        self.t_expansion = t_expansion
        self.expand_channels = t_expansion * in_channels
        with self.init_scope():
            self.pointwise_exp = Exp_Conv(t_expansion, in_channels, 1, 1)
            self.depthwise = DepthWise_Conv(self.expand_channels, self.expand_channels, ksize, stride)
            self.pointwise = PointWise_Conv(self.expand_channels, out_channels, 1, 1)


    def __call__(self, x):
        h = self.pointwise_exp(x)
        h = self.depthwise(h)
        h = self.pointwise(h)
        if x.shape == h.shape:
            return h + x
        else:
            return h


class MN_Bottleneck_expand1(chainer.link.Chain):
    def __init__(self, in_channels, out_channels, ksize, stride):
        super(MN_Bottleneck_expand1, self).__init__()
        self.expand_channels = in_channels
        with self.init_scope():
            self.depthwise = DepthWise_Conv(in_channels, self.expand_channels, ksize, stride)
            self.pointwise = PointWise_Conv(self.expand_channels, out_channels, 1, 1)

    def __call__(self, x):
        h = self.depthwise(x)
        h = self.pointwise(h)
        if x.shape == h.shape:
            return h + x
        else:
            return h



class Exp_Conv(chainer.link.Chain):
    def __init__(self, t_expansion, in_channels, ksize, stride):
        super(Exp_Conv, self).__init__()
        self.t_expansion = t_expansion
        with self.init_scope():
            self.pointwise_exp_conv = L.Convolution2D(in_channels, in_channels*t_expansion, ksize, nobias=True)
            self.pointwise_exp_bn = L.BatchNormalization(in_channels*t_expansion, eps=0.001, decay=0.997)

    def __call__(self, x):
        h = F.clipped_relu(self.pointwise_exp_bn(self.pointwise_exp_conv(x)), 6.0)
        return h


class DepthWise_Conv(chainer.link.Chain):
    def __init__(self, in_channels, out_channels, ksize, stride):
        super(DepthWise_Conv, self).__init__()
        with self.init_scope():
            self.depthwise_conv = L.Convolution2D(in_channels, out_channels, ksize, stride, ksize // 2, nobias=True
                                                  , groups=in_channels)
            self.depthwise_bn = L.BatchNormalization(out_channels, eps=0.001, decay=0.9997)

    def __call__(self, x):
        h = F.clipped_relu(self.depthwise_bn(self.depthwise_conv(x)), 6.0)
        return h


class PointWise_Conv(chainer.link.Chain):
    def __init__(self, in_channels, out_channels, ksize, stride):
        super(PointWise_Conv, self).__init__()
        with self.init_scope():
            self.pointwise_conv = L.Convolution2D(in_channels, out_channels, ksize, nobias=True)
            self.pointwise_bn = L.BatchNormalization(out_channels, eps=0.001, decay=0.9997)

    def __call__(self, x):
        h = self.pointwise_bn(self.pointwise_conv(x))
        return h


# Network definition
class MobileNetV2(chainer.Chain):
    def __init__(self, width_multiplier):
        super(MobileNetV2, self).__init__()
        with self.init_scope():
            self.conv0 = Conv(3, 32, 3, 2) #Conv2d

            self.conv1 = MN_Bottleneck_expand1(32, int(width_multiplier*16), 3, 1) # n = 1

            self.conv2 = MN_Bottleneck(6, int(width_multiplier*16), int(width_multiplier*24), 3, 2) # n = 2
            self.conv3 = MN_Bottleneck(6, int(width_multiplier*24), int(width_multiplier*24), 3, 1)

            self.conv4 = MN_Bottleneck(6, int(width_multiplier*24), int(width_multiplier*32), 3, 2) # n = 3
            self.conv5 = MN_Bottleneck(6, int(width_multiplier*32), int(width_multiplier*32), 3, 1)
            self.conv6 = MN_Bottleneck(6, int(width_multiplier*32), int(width_multiplier*32), 3, 1)

            self.conv7 = MN_Bottleneck(6, int(width_multiplier*32), int(width_multiplier*64), 3, 2) # n = 4
            self.conv8 = MN_Bottleneck(6, int(width_multiplier*64), int(width_multiplier*64), 3, 1)
            self.conv9 = MN_Bottleneck(6, int(width_multiplier*64), int(width_multiplier*64), 3, 1)
            self.conv10 = MN_Bottleneck(6, int(width_multiplier*64), int(width_multiplier*64), 3, 1)

            self.conv11 = MN_Bottleneck(6, int(width_multiplier*64), int(width_multiplier*96), 3, 1) # n = 3
            self.conv12 = MN_Bottleneck(6, int(width_multiplier*96), int(width_multiplier*96), 3, 1)
            self.conv13 = MN_Bottleneck(6, int(width_multiplier*96), int(width_multiplier*96), 3, 1)

            #self.conv14 = MN_Bottleneck(6, 96, 160, 3, 2)  # n = 3
            self.conv14_0 = Exp_Conv(6, int(width_multiplier*96), 1, 1)
            self.conv14_1 = DepthWise_Conv(int(width_multiplier*6*96), int(width_multiplier*6*96), 3, 2)
            self.conv14_2 = PointWise_Conv(int(width_multiplier*6*96), int(width_multiplier*160), 1, 1)
            self.conv15 = MN_Bottleneck(6, int(width_multiplier*160), int(width_multiplier*160), 3, 1)
            self.conv16 = MN_Bottleneck(6, int(width_multiplier*160), int(width_multiplier*160), 3, 1)

            self.conv17 = MN_Bottleneck(6, int(width_multiplier*160), int(width_multiplier*320), 3, 1)  # n = 1

            self.conv18 = Conv(int(width_multiplier*320), int(width_multiplier*1280), 1, 1) # n = 1

            #self.fc = L.Linear(None, n_class)

    def __call__(self, x):
        # Feature_map = [conv_14, conv_18, additional 4 layer]
        ys = list()
        h = self.conv0(x)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h = self.conv6(h)
        ys.append(h)
        h = self.conv7(h)
        h = self.conv8(h)
        h = self.conv9(h)
        h = self.conv10(h)
        h = self.conv11(h)
        h = self.conv12(h)
        h = self.conv13(h)
        h = self.conv14_0(h)
        ys.append(h)
        h = self.conv14_1(h)
        h = self.conv14_2(h)
        h = self.conv15(h)
        h = self.conv16(h)
        h = self.conv17(h)
        h = self.conv18(h)
        ys.append(h)
        return ys


class MobileNetV2Extractor300(chainer.Chain):
    """A MobileNet based feature extractor for SSD300.
    This is a feature extractor for :class:`~chainercv.links.model.ssd.SSD300`.
    This extractor is based on :class:`~chainercv.links.model.ssd.MobileNet`.
    """

    #insize = 300
    #grids = (19, 10, 5, 3, 2, 1) # feature map size 300x300
    #grids = (10, 10, 5, 3, 2, 1)  # feature map size 224x224

    def __init__(self, model, width_multiplier):
        super(MobileNetV2Extractor300, self).__init__()
        with self.init_scope():
            self.feature_extractor = model
            self.conv19 = FeatureMap(int(width_multiplier*1280), int(width_multiplier*256), int(width_multiplier*512))
            self.conv20 = FeatureMap(int(width_multiplier*512), int(width_multiplier*128), int(width_multiplier*256))
            self.conv21 = FeatureMap(int(width_multiplier*256), int(width_multiplier*128), int(width_multiplier*256))
            self.conv22 = FeatureMap(int(width_multiplier*256), int(width_multiplier*64), int(width_multiplier*128))

    def __call__(self, x):
        """Compute feature maps from a batch of images.
        input: (1, 3, 300, 300)
        conv0: (1, 32, 150, 150)
        conv1: (1, 16, 150, 150)
        conv2: (1, 24, 75, 75)
        conv3: (1, 24, 75, 75)
        conv4: (1, 32, 38, 38)
        conv5: (1, 32, 38, 38)
        conv6: (1, 32, 38, 38)
        conv7: (1, 64, 19, 19)
        conv8: (1, 64, 19, 19)
        conv9: (1, 64, 19, 19)
        conv10: (1, 64, 19, 19)
        conv11: (1, 96, 19, 19)
        conv12: (1, 96, 19, 19)
        conv13: (1, 96, 19, 19)
        conv14_0: (1, 576, 19, 19)
        conv14_1: (1, 576, 10, 10)
        conv14_2: (1, 160, 10, 10)
        conv15: (1, 160, 10, 10)
        conv16: (1, 160, 10, 10)
        conv17: (1, 320, 10, 10)
        conv18: (1, 1280, 10, 10)
        conv19: (1, 512, 5, 5)
        conv20: (1, 256, 3, 3)
        conv21: (1, 256, 2, 2)
        conv22: (1, 128, 1, 1)
        This method extracts feature maps from
        Args:
            x (ndarray): An array holding a batch of images.
                The images should be resized to :math:`300\\times 300`.
        Returns:
            list of Variable:
            Each variable contains a feature map.
        """

        ys = self.feature_extractor(x)
        for i in range(19, 22 + 1):
            h = ys[-1]
            h = self['conv{:d}'.format(i)](h)
            #ys.append(self['conv{:d}'.format(i)](h))
            ys.append(h)
        return ys

        # to skip unsaved parameters, use strict option.


class MobileNetV2LiteExtractor300(chainer.Chain):
    """A MobileNet based feature extractor for SSD300.
    This is a feature extractor for :class:`~chainercv.links.model.ssd.SSD300`.
    This extractor is based on :class:`~chainercv.links.model.ssd.MobileNet`.
    """

    #insize = 300
    #grids = (19, 10, 5, 3, 2, 1) # feature map size
    #grids = (10, 10, 5, 3, 2, 1)  # feature map size
    #[conv6 32/256, conv14_0 16/256, conv18 8/256, conv19 4/256, conv20 2/256, conv22 1/256]

    def __init__(self, model, width_multiplier):
        super(MobileNetV2LiteExtractor300, self).__init__()
        with self.init_scope():
            self.feature_extractor = model
            self.conv19 = FeatureMap_Lite(int(width_multiplier*1280), int(width_multiplier*256), int(width_multiplier*512))
            self.conv20 = FeatureMap_Lite(int(width_multiplier*512), int(width_multiplier*128), int(width_multiplier*256))
            self.conv21 = FeatureMap_Lite(int(width_multiplier*256), int(width_multiplier*128), int(width_multiplier*256))
            self.conv22 = FeatureMap_Lite(int(width_multiplier*256), int(width_multiplier*64), int(width_multiplier*128))

    def __call__(self, x):
        """Compute feature maps from a batch of images.
        input: (1, 3, 300, 300)
        conv0: (1, 32, 150, 150)
        conv1: (1, 16, 150, 150)
        conv2: (1, 24, 75, 75)
        conv3: (1, 24, 75, 75)
        conv4: (1, 32, 38, 38)
        conv5: (1, 32, 38, 38)
        conv6: (1, 32, 38, 38)
        conv7: (1, 64, 19, 19)
        conv8: (1, 64, 19, 19)
        conv9: (1, 64, 19, 19)
        conv10: (1, 64, 19, 19)
        conv11: (1, 96, 19, 19)
        conv12: (1, 96, 19, 19)
        conv13: (1, 96, 19, 19)
        conv14_0: (1, 576, 19, 19)
        conv14_1: (1, 576, 10, 10)
        conv14_2: (1, 160, 10, 10)
        conv15: (1, 160, 10, 10)
        conv16: (1, 160, 10, 10)
        conv17: (1, 320, 10, 10)
        conv18: (1, 1280, 10, 10)
        conv19: (1, 512, 5, 5)
        conv20: (1, 256, 3, 3)
        conv21: (1, 256, 2, 2)
        conv22: (1, 128, 1, 1)
        This method extracts feature maps from
        Args:
            x (ndarray): An array holding a batch of images.
                The images should be resized to :math:`300\\times 300`.
        Returns:
            list of Variable:
            Each variable contains a feature map.
        """

        #ys = super(MobileNetV2LiteExtractor300, self).__call__(x)
        ys = self.feature_extractor(x)
        for i in range(19, 22 + 1):
            h = ys[-1]
            h = self['conv{:d}'.format(i)](h)
            #ys.append(self['conv{:d}'.format(i)](h))
            if(i !=21):
                ys.append(h)
        return ys


if __name__ == '__main__':
    sample_array = np.ones((1, 3, 256, 256)).astype(np.float32)

    fea = MobileNetV2(width_multiplier=1.0)
    model = MobileNetV2LiteExtractor300(fea, width_multiplier=1.0)
    #model.add_insize_stride(300, (19, 10, 5, 3, 2, 1))
    ys = model(sample_array)
    #print(model.insize)
    print(ys[0].shape)

    #input (1, 3, 256, 256)
    #conv_0(1, 32, 128, 128)
    #conv_1(1, 16, 128, 128)
    #conv_2(1, 24, 64, 64)
    #conv_3(1, 24, 64, 64)
    #conv_4(1, 32, 32, 32)
    #conv_5(1, 32, 32, 32)
    #conv_6(1, 32, 32, 32)
    #conv_7(1, 64, 16, 16)
    #conv_8(1, 64, 16, 16)
    #conv_9(1, 64, 16, 16)
    #conv_10(1, 64, 16, 16)
    #conv_11(1, 96, 16, 16)
    #conv_12(1, 96, 16, 16)
    #conv_13(1, 96, 16, 16)
    #conv_14_0(1, 576, 16, 16)
    #conv_14_1(1, 576, 8, 8)
    #conv_14_2(1, 160, 8, 8)
    #conv_15(1, 160, 8, 8)
    #conv_16(1, 160, 8, 8)
    #conv_17(1, 320, 8, 8)
    #conv_18(1, 1280, 8, 8)
    #conv19(1, 512, 4, 4)
    #conv20(1, 256, 2, 2)
    #conv21(1, 256, 1, 1)
    #conv22(1, 128, 1, 1)
