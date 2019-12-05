# -*- coding: utf-8 -*-

import numpy as np
import chainer
import math


def generate_anchor_box_size(input_size, smin, smax, num_layers):
    # Add 1.0 to the end, which will only be used in scale_next below and used
    # for computing an interpolated scale for the largest scale in the list.
    scales = [smin + (smax - smin) * i / (num_layers - 1) for i in range(num_layers)]
    scales = scales + [1.0]
    scales = [math.ceil(input_size * x) for x in scales]
    return scales


def generate_anchor_stride(input_size, feature_map_list):
    stride = [math.ceil(input_size / x[0]) for x in feature_map_list]
    return stride


def get_feature_map_layout(input_size, extractor):
    sample_array = np.ones((1, 3, input_size, input_size)).astype(np.float32)
    with chainer.using_config('train', False):
        ys = extractor(sample_array)

    feature_map_layout = []
    for fmap in ys:
        fmap_shape = fmap.shape
        feature_map_layout.append((fmap_shape[2], fmap_shape[3]))
    return feature_map_layout


def get_ssd_extractor_channel(input_size, extractor):
    sample_array = np.ones((1, 3, input_size, input_size)).astype(np.float32)
    with chainer.using_config('train', False):
        ys = extractor(sample_array)

    channel_layout = []
    for ch in ys:
        channel_layout.append(ch.shape[1])
    return channel_layout
