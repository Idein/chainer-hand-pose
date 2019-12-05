import numpy as np

# Utilities taken from ChainerCV project
# non_maximum_suppression: https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/non_maximum_suppression.py
# resize_bbox: https://github.com/chainer/chainercv/blob/master/chainercv/transforms/bbox/resize_bbox.py
# flip_point: https://github.com/chainer/chainercv/blob/master/chainercv/transforms/point/flip_point.py


def flip_point(point, size, y_flip=False, x_flip=False):
    """Modify points according to image flips.
    Args:
        point (~numpy.ndarray or list of arrays): See the table below.
        size (tuple): A tuple of length 2. The height and the width
            of the image, which is associated with the points.
        y_flip (bool): Modify points according to a vertical flip of
            an image.
        x_flip (bool): Modify keypoipoints according to a horizontal flip of
            an image.
    .. csv-table::
        :header: name, shape, dtype, format
        :obj:`point`, ":math:`(R, K, 2)` or :math:`[(K, 2)]`", \
        :obj:`float32`, ":math:`(y, x)`"
    Returns:
        ~numpy.ndarray or list of arrays:
        Points modified according to image flips.
    """
    H, W = size
    if isinstance(point, np.ndarray):
        out_point = point.copy()
        if y_flip:
            out_point[:, :, 0] = H - out_point[:, :, 0]
        if x_flip:
            out_point[:, :, 1] = W - out_point[:, :, 1]
    else:
        out_point = []
        for pnt in point:
            pnt = pnt.copy()
            if y_flip:
                pnt[:, 0] = H - pnt[:, 0]
            if x_flip:
                pnt[:, 1] = W - pnt[:, 1]
            out_point.append(pnt)
    return out_point


def non_maximum_suppression(bbox, thresh, score=None, limit=None):
    """Suppress bounding boxes according to their IoUs.
        This method checks each bounding box sequentially and selects the bounding
        box if the Intersection over Unions (IoUs) between the bounding box and the
        previously selected bounding boxes is less than :obj:`thresh`. This method
        is mainly used as postprocessing of object detection.
        The bounding boxes are selected from ones with higher scores.
        If :obj:`score` is not provided as an argument, the bounding box
        is ordered by its index in ascending order.
        The bounding boxes are expected to be packed into a two dimensional
        tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
        bounding boxes in the image. The second axis represents attributes of
        the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
        where the four attributes are coordinates of the top left and the
        bottom right vertices.
        :obj:`score` is a float array of shape :math:`(R,)`. Each score indicates
        confidence of prediction.
        This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
        an input. Please note that both :obj:`bbox` and :obj:`score` need to be
        the same type.
        The type of the output is the same as the input.
        Args:
            bbox (array): Bounding boxes to be transformed. The shape is
                :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
            thresh (float): Threshold of IoUs.
            score (array): An array of confidences whose shape is :math:`(R,)`.
            limit (int): The upper bound of the number of the output bounding
                boxes. If it is not specified, this method selects as many
                bounding boxes as possible.
        Returns:
            array:
            An array with indices of bounding boxes that are selected. \
            They are sorted by the scores of bounding boxes in descending \
            order. \
            The shape of this array is :math:`(K,)` and its dtype is\
            :obj:`numpy.int32`. Note that :math:`K \\leq R`.
        """
    if len(bbox) == 0:
        return np.zeros((0,), dtype=np.int32)

    if score is not None:
        order = score.argsort()[::-1]
        bbox = bbox[order]
    bbox_area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)

    selec = np.zeros(bbox.shape[0], dtype=bool)
    for i, b in enumerate(bbox):
        tl = np.maximum(b[:2], bbox[selec, :2])
        br = np.minimum(b[2:], bbox[selec, 2:])
        area = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)

        iou = area / (bbox_area[i] + bbox_area[selec] - area)
        if (iou >= thresh).any():
            continue

        selec[i] = True
        if limit is not None and np.count_nonzero(selec) >= limit:
            break

    selec = np.where(selec)[0]
    if score is not None:
        selec = order[selec]
    return selec.astype(np.int32)


def resize_bbox(bbox, in_size, out_size):
    """Resize bounding boxes according to image resize.
    Args:
        bbox (~numpy.ndarray): See the table below.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.
    .. csv-table::
        :header: name, shape, dtype, format
        :obj:`bbox`, ":math:`(R, 4)`", :obj:`float32`, \
        ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"
    Returns:
        ~numpy.ndarray:
        Bounding boxes rescaled according to the given image shapes.
    """
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox
