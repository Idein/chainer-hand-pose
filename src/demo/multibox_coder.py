import itertools

import numpy as np

from transforms import non_maximum_suppression


class MultiBoxCoder():
    """
    partially taken from src/detector/model/multibox_decoder
    """

    def __init__(self, grids, aspect_ratios, steps, sizes, variance=(0.1, 0.2)):
        if not len(aspect_ratios) == len(grids):
            raise ValueError('The length of aspect_ratios is wrong.')
        if not len(steps) == len(grids):
            raise ValueError('The length of steps is wrong.')
        if not len(sizes) == len(grids) + 1:
            raise ValueError('The length of sizes is wrong.')

        default_bbox = list()

        for k, grid in enumerate(grids):
            for v, u in itertools.product(range(grid), repeat=2):
                cy = (v + 0.5) * steps[k]
                cx = (u + 0.5) * steps[k]

                s = sizes[k]
                default_bbox.append((cy, cx, s, s))

                s = np.sqrt(sizes[k] * sizes[k + 1])
                default_bbox.append((cy, cx, s, s))

                #s = sizes[k]
                # for ar in aspect_ratios[k]:
                #    default_bbox.append(
                #        (cy, cx, s / np.sqrt(ar), s * np.sqrt(ar)))
                #    default_bbox.append(
                #        (cy, cx, s * np.sqrt(ar), s / np.sqrt(ar)))

        # (center_y, center_x, height, width)
        self._default_bbox = np.stack(default_bbox)
        self._variance = variance

    def decode(self, mb_loc, mb_conf, nms_thresh=0.45, score_thresh=0.6):
        """Decodes back to coordinates and classes of bounding boxes.
        This method decodes :obj:`mb_loc` and :obj:`mb_conf` returned
        by a SSD network back to :obj:`bbox`, :obj:`label` and :obj:`score`.
        Args:
            mb_loc (array): A float array whose shape is
                :math:`(K, 4)`, :math:`K` is the number of
                default bounding boxes.
            mb_conf (array): A float array whose shape is
                :math:`(K, n\_fg\_class + 1)`.
            nms_thresh (float): The threshold value
                for :func:`~chainercv.utils.non_maximum_suppression`.
                The default value is :obj:`0.45`.
            score_thresh (float): The threshold value for confidence score.
                If a bounding box whose confidence score is lower than
                this value, the bounding box will be suppressed.
                The default value is :obj:`0.6`.
        Returns:
            tuple of three arrays:
            This method returns a tuple of three arrays,
            :obj:`(bbox, label, score)`.
            * **bbox**: A float array of shape :math:`(R, 4)`, \
                where :math:`R` is the number of bounding boxes in a image. \
                Each bouding box is organized by \
                :math:`(y_{min}, x_{min}, y_{max}, x_{max})` \
                in the second axis.
            * **label** : An integer array of shape :math:`(R,)`. \
                Each value indicates the class of the bounding box.
            * **score** : A float array of shape :math:`(R,)`. \
                Each value indicates how confident the prediction is.
        """

        # (center_y, center_x, height, width)
        mb_bbox = self._default_bbox.copy()
        mb_bbox[:, :2] += mb_loc[:, :2] * self._variance[0] \
            * self._default_bbox[:, 2:]
        mb_bbox[:, 2:] *= np.exp(mb_loc[:, 2:] * self._variance[1])

        # (center_y, center_x, height, width) -> (y_min, x_min, height, width)
        mb_bbox[:, :2] -= mb_bbox[:, 2:] / 2
        # (center_y, center_x, height, width) -> (y_min, x_min, y_max, x_max)
        mb_bbox[:, 2:] += mb_bbox[:, :2]

        # softmax
        mb_score = np.exp(mb_conf)
        mb_score /= mb_score.sum(axis=1, keepdims=True)

        bbox = list()
        label = list()
        score = list()
        for l in range(mb_conf.shape[1] - 1):
            bbox_l = mb_bbox
            # the l-th class corresponds for the (l + 1)-th column.
            score_l = mb_score[:, l + 1]

            mask = score_l >= score_thresh
            bbox_l = bbox_l[mask]
            score_l = score_l[mask]

            if nms_thresh is not None:
                indices = non_maximum_suppression(
                    bbox_l, nms_thresh, score_l)
                bbox_l = bbox_l[indices]
                score_l = score_l[indices]

            bbox.append(bbox_l)
            label.append(np.array((l,) * len(bbox_l)))
            score.append(score_l)

        bbox = np.vstack(bbox).astype(np.float32)
        label = np.hstack(label).astype(np.int32)
        score = np.hstack(score).astype(np.float32)

        return bbox, label, score
