from __future__ import division

import numpy as np

import chainer
import chainer.functions as F



def multibox_focal_loss(mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, k):
    """Computes multibox losses.
    This is a loss function used in [#]_.
    This function returns :obj:`loc_loss` and :obj:`conf_loss`.
    :obj:`loc_loss` is a loss for localization and
    :obj:`conf_loss` is a loss for classification.
    The formulas of these losses can be found in
    the equation (2) and (3) in the original paper.
    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan,
       Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.
    Args:
        mb_locs (chainer.Variable or array): The offsets and scales
            for predicted bounding boxes.
            Its shape is :math:`(B, K, 4)`,
            where :math:`B` is the number of samples in the batch and
            :math:`K` is the number of default bounding boxes.
        mb_confs (chainer.Variable or array): The classes of predicted
            bounding boxes.
            Its shape is :math:`(B, K, n\_class)`.
            This function assumes the first class is background (negative).
        gt_mb_locs (chainer.Variable or array): The offsets and scales
            for ground truth bounding boxes.
            Its shape is :math:`(B, K, 4)`.
        gt_mb_labels (chainer.Variable or array): The classes of ground truth
            bounding boxes.
            Its shape is :math:`(B, K)`.
        k (float): A coefficient which is used for hard negative mining.
            This value determines the ratio between the number of positives
            and that of mined negatives. The value used in the original paper
            is :obj:`3`.
    Returns:
        tuple of chainer.Variable:
        This function returns two :obj:`chainer.Variable`: :obj:`loc_loss` and
        :obj:`conf_loss`.
    """
    mb_locs = chainer.as_variable(mb_locs)
    mb_confs = chainer.as_variable(mb_confs)
    gt_mb_locs = chainer.as_variable(gt_mb_locs)
    #gt_mb_labels = chainer.as_variable(gt_mb_labels)

    xp = chainer.cuda.get_array_module(gt_mb_locs.array)

    #print(gt_mb_labels.array.device)
    #print('Multibox')
    #print(chainer.cuda.get_device_from_array(gt_mb_labels.array))

    #with gt_mb_labels.array.device:
    #positive = gt_mb_labels.array > 0
    positive = gt_mb_labels > 0
    n_positive = positive.sum()

    if n_positive == 0:
        z = chainer.Variable(xp.zeros((), dtype=np.float32))
        return z, z

    loc_loss = F.huber_loss(mb_locs, gt_mb_locs, 1, reduce='no')
    loc_loss = F.sum(loc_loss, axis=-1)
    loc_loss *= positive.astype(loc_loss.dtype)
    loc_loss = F.sum(loc_loss) / n_positive

    #conf_loss = _elementwise_softmax_cross_entropy(mb_confs, gt_mb_labels)
    #hard_negative = _hard_negative(conf_loss.array, positive, k)
    #conf_loss *= xp.logical_or(positive, hard_negative).astype(conf_loss.dtype)

    alpha = 0.75
    gamma = 2

    t = gt_mb_labels.reshape(gt_mb_labels.shape[0]*gt_mb_labels.shape[1], )
    class_num = mb_confs.shape[2] # class_num includes back ground class
    t = F.cast(chainer.as_variable(xp.eye(class_num)[t]), loc_loss.dtype)
    t = t.reshape(gt_mb_labels.shape[0], gt_mb_labels.shape[1], class_num)

    p =  F.sigmoid(mb_confs)
    #pt = p * t + (1 - p) * (1 - t) # pt = p if t > 0 else 1-p
    #w = alpha * t + (1 - alpha) * (1 - t)  # w = alpha if t > 0 else 1 - alpha
    #w = w * ((1 - pt) ** gamma)

    pt = F.where(t.array > 0, p, 1-p)
    w = (1 - pt) ** gamma
    w = F.where(t.array > 0, alpha * w, (1 - alpha) * w)

    # From Pytorch implemetation binary_cross_entropy_with_logits
    # https://pytorch.org/docs/master/_modules/torch/nn/functional.html#binary_cross_entropy_with_logits
    max_val = F.clip(-mb_confs, x_min=0.0, x_max=10.0e+12)
    focal_loss = mb_confs - mb_confs * t + max_val + F.log(F.exp(-max_val) + F.exp(-mb_confs - max_val))
    focal_loss = F.sum(focal_loss * w) / n_positive
    #focal_loss = -F.sum(w * F.log(pt + 1e-12)) / n_positive

    return loc_loss, focal_loss
