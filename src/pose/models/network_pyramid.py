import chainer
from pose.models.network_base import Convolution2d, Conv1x1
from pose.models.network_base import SeparableConv, ExpandedConv
import chainer.functions as F


class GlobalNet(chainer.Chain):
    """
    inspired by CascadedPyramidNetwork
    """

    def __init__(self, pose_multiplier=1.0):
        super(GlobalNet, self).__init__()
        min_depth = 8

        def pose(d): return max(int(d * pose_multiplier), min_depth)

        with self.init_scope():
            self.lateral_bottom = Conv1x1(None, pose(128), relu=True)
            self.merge_bottom = Conv1x1(pose(128), pose(128), relu=False)

            self.lateral_middle = Conv1x1(None, pose(128), relu=True)
            self.merge_middle = Conv1x1(pose(128), pose(128), relu=False)

            self.lateral_top = Conv1x1(None, pose(128), relu=True)

    def __call__(self, top, middle, bottom):
        # bottom
        lateral_bottom = self.lateral_bottom(bottom)
        upsample = F.resize_images(lateral_bottom,
                                   (middle.shape[2], middle.shape[3]))
        merge_bottom = self.merge_bottom(upsample)
        # middle
        lateral_middle = self.lateral_middle(middle)
        last_fm_bottom_middle = merge_bottom + lateral_middle
        upsample = F.resize_images(last_fm_bottom_middle,
                                   (top.shape[2], top.shape[3]))
        merge_middle = self.merge_middle(upsample)
        # top
        lateral_top = self.lateral_top(top)
        last_fm_middle_top = merge_middle + lateral_top

        return last_fm_middle_top, last_fm_bottom_middle, lateral_bottom


class GlobalLoss(chainer.Chain):

    def __init__(self, pose_multiplier=1.0, upsample=False):
        super(GlobalLoss, self).__init__()
        self.upsample = upsample
        min_depth = 8

        def pose(d): return max(int(d * pose_multiplier), min_depth)

        with self.init_scope():
            # pose layer vect
            self.vect_conv1 = SeparableConv(None, pose(64))
            self.vect_conv2 = SeparableConv(pose(64), pose(64))
            self.vect_conv3 = SeparableConv(pose(64), pose(64))
            self.vect_conv4 = SeparableConv(pose(64), pose(64), ksize=1)

            # pose layer heat
            self.heat_conv1 = SeparableConv(None, pose(64))
            self.heat_conv2 = SeparableConv(pose(64), pose(64))
            self.heat_conv3 = SeparableConv(pose(64), pose(64))
            self.heat_conv4 = SeparableConv(pose(64), pose(64), ksize=1)

    def __call__(self, last_fm_middle_top):

        h = self.vect_conv1(last_fm_middle_top)
        h = self.vect_conv2(h)
        if self.upsample:
            h = F.resize_images(h, (2 * h.shape[2], 2 * h.shape[3]))
        h = self.vect_conv3(h)
        vect_out = self.vect_conv4(h)

        h = self.heat_conv1(last_fm_middle_top)
        h = self.heat_conv2(h)
        if self.upsample:
            h = F.resize_images(h, (2 * h.shape[2], 2 * h.shape[3]))
        h = self.heat_conv3(h)
        heat_out = self.heat_conv4(h)
        return vect_out, heat_out


class RefineLoss(chainer.Chain):

    def __init__(self, pose_multiplier=1.0, upsample=False):
        super(RefineLoss, self).__init__()
        min_depth = 8

        def pose(d): return max(int(d * pose_multiplier), min_depth)

        self.upsample = upsample
        with self.init_scope():
            # pose layer vect
            self.vect_conv1 = SeparableConv(None, pose(64))
            self.vect_conv2 = SeparableConv(pose(64), pose(64))
            self.vect_conv3 = SeparableConv(pose(64), pose(64))
            self.vect_conv4 = SeparableConv(pose(64), pose(64), ksize=1)

            # pose layer heat
            self.heat_conv1 = SeparableConv(None, pose(64))
            self.heat_conv2 = SeparableConv(pose(64), pose(64))
            self.heat_conv3 = SeparableConv(pose(64), pose(64))
            self.heat_conv4 = SeparableConv(pose(64), pose(64), ksize=1)

    def __call__(self, refine_concat):
        h = self.vect_conv1(refine_concat)
        h = self.vect_conv2(h)
        if self.upsample:
            h = F.resize_images(h, (2 * h.shape[2], 2 * h.shape[3]))
        h = self.vect_conv3(h)
        vect_out = self.vect_conv4(h)

        h = self.heat_conv1(refine_concat)
        h = self.heat_conv2(h)
        if self.upsample:
            h = F.resize_images(h, (2 * h.shape[2], 2 * h.shape[3]))
        h = self.heat_conv3(h)
        heat_out = F.sigmoid(self.heat_conv4(h))
        return vect_out, heat_out


class RefineNet(chainer.Chain):
    """
    inspired by CascadedPyramidNetwork
    """

    def __init__(self, pose_multiplier=1.0):
        super(RefineNet, self).__init__()
        min_depth = 8

        def pose(d): return max(int(d * pose_multiplier), min_depth)

        with self.init_scope():
            # bottom
            self.refine_1_1 = SeparableConv(None, pose(128))
            self.refine_1_2 = SeparableConv(pose(128), pose(128))
            self.refine_1_3 = SeparableConv(pose(128), pose(128))
            # middle
            self.refine_2_1 = SeparableConv(None, pose(128))
            self.refine_2_2 = SeparableConv(pose(128), pose(128))
            # top
            self.refine_3_1 = SeparableConv(None, pose(128))

    def __call__(self, top, middle, bottom):
        h = self.refine_1_1(bottom)
        h = self.refine_1_2(h)
        h = self.refine_1_3(h)
        refine_1_upsample = F.resize_images(h, (top.shape[2], top.shape[3]))
        h = self.refine_2_1(middle)
        h = self.refine_2_2(h)
        refine_2_upsample = F.resize_images(h, (top.shape[2], top.shape[3]))
        h_top = self.refine_3_1(top)
        refine_concat = F.concat(
            (refine_1_upsample,
             refine_2_upsample,
             h_top
             ),
            axis=1
        )
        return refine_concat


class MobileNetV2(chainer.Chain):
    """
    custum network based on MobileNetV2
    """

    def __init__(self, width_multiplier=1.0):
        super(MobileNetV2, self).__init__()
        min_depth = 8

        def multiplier(d): return max(int(d * width_multiplier), min_depth)

        with self.init_scope():
            self.conv0 = Convolution2d(None, multiplier(32), stride=2)
            self.conv1 = ExpandedConv(1, multiplier(32), multiplier(16), stride=1)
            self.conv2 = ExpandedConv(6, multiplier(16), multiplier(24), stride=2)
            self.conv3 = ExpandedConv(6, multiplier(24), multiplier(24), stride=1)
            self.conv4 = ExpandedConv(6, multiplier(24), multiplier(32), stride=2)
            self.conv5 = ExpandedConv(6, multiplier(32), multiplier(32), stride=1)
            self.conv6 = ExpandedConv(6, multiplier(32), multiplier(32), stride=1)
            self.conv7 = ExpandedConv(6, multiplier(32), multiplier(64), stride=2)
            self.conv8 = ExpandedConv(6, multiplier(64), multiplier(64), stride=1)
            self.conv9 = ExpandedConv(6, multiplier(64), multiplier(64), stride=1)
            self.conv10 = ExpandedConv(6, multiplier(64), multiplier(64), stride=1)
            # modify conv11 stride 1 -> 2
            self.conv11 = ExpandedConv(6, multiplier(64), multiplier(96), stride=2)
            self.conv12 = ExpandedConv(6, multiplier(96), multiplier(96), stride=1)
            self.conv13 = ExpandedConv(6, multiplier(96), multiplier(96), stride=1)
            # self.conv14 = ExpandedConv(6, multiplier(96), multiplier(160), stride=2)
            # self.conv15 = ExpandedConv(6, multiplier(160), multiplier(160), stride=1)
            # self.conv16 = ExpandedConv(6, multiplier(160), multiplier(160), stride=1)
            # self.conv17 = ExpandedConv(6, multiplier(160), multiplier(320), stride=1)

    def __call__(self, x):
        features = []  # h_top, h_middle, h_bottom
        top = 6
        middle = 10
        bottom = 13
        h = self.conv0(x)
        for i in range(1, 13 + 1):
            conv = eval("self.conv{}".format(i))
            h = conv(h)
            if i in [top, middle, bottom]:
                features.append(h)
        # h_top, h_middle, h_bottom
        return features

    """

    def __call__(self, x):
        h = self.conv0(x)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h_top = self.conv6(h)
        h = self.conv7(h_top)
        h = self.conv8(h)
        h = self.conv9(h)
        h_middle = self.conv10(h)
        h = self.conv11(h)
        h = self.conv12(h)
        h_bottom = self.conv13(h)
        return h_top, h_middle, h_bottom
    """


class CascadedPyramidNetwork(chainer.Chain):
    """
    Pose estimation model its base network is MobileNetV2
    """

    def __init__(self,
                 num_heat_ch,
                 num_vect_ch,
                 pose_multiplier=1.0,
                 width_multiplier=1.0,
                 **kwargs):
        super(CascadedPyramidNetwork, self).__init__()
        with self.init_scope():
            self.mobilenetv2 = MobileNetV2(width_multiplier)
            self.globalnet = GlobalNet(pose_multiplier)
            self.refinenet = RefineNet(pose_multiplier)
            self.refineloss = RefineLoss(pose_multiplier, upsample=True)
            self.globalloss = GlobalLoss(pose_multiplier, upsample=True)

            self.global_vect_convout = SeparableConv(None, num_vect_ch,
                                                     ksize=1, relu=False)
            self.global_heat_convout = SeparableConv(None, num_heat_ch,
                                                     ksize=1, relu=False)

            self.refine_vect_convout = SeparableConv(None, num_vect_ch,
                                                     ksize=1, relu=False)
            self.refine_heat_convout = SeparableConv(None, num_heat_ch,
                                                     ksize=1, relu=False)

    def __call__(self, x):
        top, middle, bottom = self.mobilenetv2(x)
        top, middle, bottom = self.globalnet(top, middle, bottom)
        global_vect, global_heat = self.globalloss(top)
        global_vect = self.global_vect_convout(global_vect)
        global_heat = self.global_heat_convout(global_heat)
        refine_concat = self.refinenet(top, middle, bottom)
        refine_vect, refine_heat = self.refineloss(refine_concat)
        refine_vect = self.refine_vect_convout(refine_vect)
        refine_heat = self.refine_heat_convout(refine_heat)
        global_heat = F.clipped_relu(global_heat, 1.1)
        refine_heat = F.clipped_relu(refine_heat, 1.1)
        return [global_vect, refine_vect], [global_heat, refine_heat]

    @staticmethod
    def prepare(x):
        return x / 255


def main():
    import numpy as np
    model = CascadedPyramidNetwork(16, 17 * 3)
    inp = np.ones((2, 3, 256, 256)).astype(np.float32)
    vect, heat = model(inp)
    print(vect)


if __name__ == '__main__':
    main()
