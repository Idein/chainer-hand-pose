from __future__ import division

import numpy as np
import random
import six

from chainercv import utils


def random_distort(
        img,
        brightness_delta=32,
        contrast_low=0.5, contrast_high=1.5,
        saturation_low=0.5, saturation_high=1.5,
        hue_delta=18):
    """A color related data augmentation used in SSD.
    This function is a combination of four augmentation methods:
    brightness, contrast, saturation and hue.
    * brightness: Adding a random offset to the intensity of the image.
    * contrast: Multiplying the intensity of the image by a random scale.
    * saturation: Multiplying the saturation of the image by a random scale.
    * hue: Adding a random offset to the hue of the image randomly.
    This data augmentation is used in training of
    Single Shot Multibox Detector [#]_.
    Note that this function requires :mod:`cv2`.
    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.
    Args:
        img (~numpy.ndarray): An image array to be augmented. This is in
            CHW and RGB format.
        brightness_delta (float): The offset for saturation will be
            drawn from :math:`[-brightness\_delta, brightness\_delta]`.
            The default value is :obj:`32`.
        contrast_low (float): The scale for contrast will be
            drawn from :math:`[contrast\_low, contrast\_high]`.
            The default value is :obj:`0.5`.
        contrast_high (float): See :obj:`contrast_low`.
            The default value is :obj:`1.5`.
        saturation_low (float): The scale for saturation will be
            drawn from :math:`[saturation\_low, saturation\_high]`.
            The default value is :obj:`0.5`.
        saturation_high (float): See :obj:`saturation_low`.
            The default value is :obj:`1.5`.
        hue_delta (float): The offset for hue will be
            drawn from :math:`[-hue\_delta, hue\_delta]`.
            The default value is :obj:`18`.
    Returns:
        An image in CHW and RGB format.
    """
    import cv2

    cv_img = img[::-1].transpose((1, 2, 0)).astype(np.uint8)

    def convert(img, alpha=1, beta=0):
        img = img.astype(float) * alpha + beta
        img[img < 0] = 0
        img[img > 255] = 255
        return img.astype(np.uint8)

    def brightness(cv_img, delta):
        if random.randrange(2):
            return convert(
                cv_img,
                beta=random.uniform(-delta, delta))
        else:
            return cv_img

    def contrast(cv_img, low, high):
        if random.randrange(2):
            return convert(
                cv_img,
                alpha=random.uniform(low, high))
        else:
            return cv_img

    def saturation(cv_img, low, high):
        if random.randrange(2):
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
            cv_img[:, :, 1] = convert(
                cv_img[:, :, 1],
                alpha=random.uniform(low, high))
            return cv2.cvtColor(cv_img, cv2.COLOR_HSV2BGR)
        else:
            return cv_img

    def hue(cv_img, delta):
        if random.randrange(2):
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
            cv_img[:, :, 0] = (
                cv_img[:, :, 0].astype(int) +
                random.randint(-delta, delta)) % 180
            return cv2.cvtColor(cv_img, cv2.COLOR_HSV2BGR)
        else:
            return cv_img

    cv_img = brightness(cv_img, brightness_delta)

    if random.randrange(2):
        cv_img = contrast(cv_img, contrast_low, contrast_high)
        cv_img = saturation(cv_img, saturation_low, saturation_high)
        cv_img = hue(cv_img, hue_delta)
    else:
        cv_img = saturation(cv_img, saturation_low, saturation_high)
        cv_img = hue(cv_img, hue_delta)
        cv_img = contrast(cv_img, contrast_low, contrast_high)

    return cv_img.astype(np.float32).transpose((2, 0, 1))[::-1]


def random_crop_with_bbox_constraints(
        img, bbox, min_scale=0.3, max_scale=1,
        max_aspect_ratio=2, constraints=None,
        max_trial=50, return_param=False):
    """Crop an image randomly with bounding box constraints.
    This data augmentation is used in training of
    Single Shot Multibox Detector [#]_. More details can be found in
    data augmentation section of the original paper.
    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.
    Args:
        img (~numpy.ndarray): An image array to be cropped. This is in
            CHW format.
        bbox (~numpy.ndarray): Bounding boxes used for constraints.
            The shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        min_scale (float): The minimum ratio between a cropped
            region and the original image. The default value is :obj:`0.3`.
        max_scale (float): The maximum ratio between a cropped
            region and the original image. The default value is :obj:`1`.
        max_aspect_ratio (float): The maximum aspect ratio of cropped region.
            The default value is :obj:`2`.
        constaraints (iterable of tuples): An iterable of constraints.
            Each constraint should be :obj:`(min_iou, max_iou)` format.
            If you set :obj:`min_iou` or :obj:`max_iou` to :obj:`None`,
            it means not limited.
            If this argument is not specified, :obj:`((0.1, None), (0.3, None),
            (0.5, None), (0.7, None), (0.9, None), (None, 1))` will be used.
        max_trial (int): The maximum number of trials to be conducted
            for each constraint. If this function
            can not find any region that satisfies the constraint in
            :math:`max\_trial` trials, this function skips the constraint.
            The default value is :obj:`50`.
        return_param (bool): If :obj:`True`, this function returns
            information of intermediate values.
    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):
        If :obj:`return_param = False`,
        returns an array :obj:`img` that is cropped from the input
        array.
        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`img, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.
        * **constraint** (*tuple*): The chosen constraint.
        * **y_slice** (*slice*): A slice in vertical direction used to crop \
            the input image.
        * **x_slice** (*slice*): A slice in horizontal direction used to crop \
            the input image.
    """

    if constraints is None:
        constraints = (
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, 1),
        )

    _, H, W = img.shape
    params = [{
        'constraint': None, 'y_slice': slice(0, H), 'x_slice': slice(0, W)}]

    if len(bbox) == 0:
        constraints = list()

    for min_iou, max_iou in constraints:
        if min_iou is None:
            min_iou = 0
        if max_iou is None:
            max_iou = 1

        for _ in six.moves.range(max_trial):
            scale = random.uniform(min_scale, max_scale)
            aspect_ratio = random.uniform(
                max(1 / max_aspect_ratio, scale * scale),
                min(max_aspect_ratio, 1 / (scale * scale)))
            crop_h = int(H * scale / np.sqrt(aspect_ratio))
            crop_w = int(W * scale * np.sqrt(aspect_ratio))

            crop_t = random.randrange(H - crop_h)
            crop_l = random.randrange(W - crop_w)
            crop_bb = np.array((
                crop_t, crop_l, crop_t + crop_h, crop_l + crop_w))

            iou = utils.bbox_iou(bbox, crop_bb[np.newaxis])
            if min_iou <= iou.min() and iou.max() <= max_iou:
                params.append({
                    'constraint': (min_iou, max_iou),
                    'y_slice': slice(crop_t, crop_t + crop_h),
                    'x_slice': slice(crop_l, crop_l + crop_w)})
                break

    param = random.choice(params)
    img = img[:, param['y_slice'], param['x_slice']]

    if return_param:
        return img, param
    else:
        return img


def resize_with_random_interpolation(img, size, return_param=False):
    """Resize an image with a randomly selected interpolation method.
    This function is similar to :func:`chainercv.transforms.resize`, but
    this chooses the interpolation method randomly.
    This data augmentation is used in training of
    Single Shot Multibox Detector [#]_.
    Note that this function requires :mod:`cv2`.
    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.
    Args:
        img (~numpy.ndarray): An array to be transformed.
            This is in CHW format and the type should be :obj:`numpy.float32`.
        size (tuple): This is a tuple of length 2. Its elements are
            ordered as (height, width).
        return_param (bool): Returns information of interpolation.
    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):
        If :obj:`return_param = False`,
        returns an array :obj:`img` that is the result of rotation.
        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`img, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.
        * **interpolatation**: The chosen interpolation method.
    """

    import cv2

    cv_img = img.transpose((1, 2, 0))

    inters = (
        cv2.INTER_LINEAR,
        cv2.INTER_AREA,
        cv2.INTER_NEAREST,
        cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4,
    )
    inter = random.choice(inters)
    H, W = size
    cv_img = cv2.resize(cv_img, (W, H), interpolation=inter)

    # If input is a grayscale image, cv2 returns a two-dimentional array.
    if len(cv_img.shape) == 2:
        cv_img = cv_img[:, :, np.newaxis]

    img = cv_img.astype(np.float32).transpose((2, 0, 1))

    if return_param:
        return img, {'interpolation': inter}
    else:
        return img


def random_expand(img, max_ratio=4, fill=0, return_param=False):
    """Expand an image randomly.
    This method randomly place the input image on a larger canvas. The size of
    the canvas is :math:`(rH, rW)`, where :math:`(H, W)` is the size of the
    input image and :math:`r` is a random ratio drawn from
    :math:`[1, max\_ratio]`. The canvas is filled by a value :obj:`fill`
    except for the region where the original image is placed.
    This data augmentation trick is used to create "zoom out" effect [#]_.
    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, \
    Scott Reed, Cheng-Yang Fu, Alexander C. Berg. \
    SSD: Single Shot MultiBox Detector. ECCV 2016.
    Args:
        img (~numpy.ndarray): An image array to be augmented. This is in
            CHW format.
        max_ratio (float): The maximum ratio of expansion. In the original
            paper, this value is 4.
        fill (float, tuple or ~numpy.ndarray): The value of padded pixels.
            In the original paper, this value is the mean of ImageNet.
            If it is :class:`numpy.ndarray`,
            its shape should be :math:`(C, 1, 1)`,
            where :math:`C` is the number of channels of :obj:`img`.
        return_param (bool): Returns random parameters.
    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):
        If :obj:`return_param = False`,
        returns an array :obj:`out_img` that is the result of expansion.
        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`out_img, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.
        * **ratio** (*float*): The sampled value used to make the canvas.
        * **y_offset** (*int*): The y coodinate of the top left corner of\
            the image after placing on the canvas.
        * **x_offset** (*int*): The x coordinate of the top left corner\
            of the image after placing on the canvas.
    """

    if max_ratio <= 1:
        if return_param:
            return img, {'ratio': 1, 'y_offset': 0, 'x_offset': 0}
        else:
            return img

    C, H, W = img.shape

    ratio = random.uniform(1, max_ratio)
    out_H, out_W = int(H * ratio), int(W * ratio)

    y_offset = random.randint(0, out_H - H)
    x_offset = random.randint(0, out_W - W)

    out_img = np.empty((C, out_H, out_W), dtype=img.dtype)
    out_img[:] = np.array(fill).reshape((-1, 1, 1))
    out_img[:, y_offset:y_offset + H, x_offset:x_offset + W] = img

    if return_param:
        param = {'ratio': ratio, 'y_offset': y_offset, 'x_offset': x_offset}
        return out_img, param
    else:
        return out_img


def translate_bbox(bbox, y_offset=0, x_offset=0):
    """Translate bounding boxes.
    This method is mainly used together with image transforms, such as padding
    and cropping, which translates the left top point of the image from
    coordinate :math:`(0, 0)` to coordinate
    :math:`(y, x) = (y_{offset}, x_{offset})`.
    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.
    Args:
        bbox (~numpy.ndarray): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        y_offset (int or float): The offset along y axis.
        x_offset (int or float): The offset along x axis.
    Returns:
        ~numpy.ndarray:
        Bounding boxes translated according to the given offsets.
    """

    out_bbox = bbox.copy()
    out_bbox[:, :2] += (y_offset, x_offset)
    out_bbox[:, 2:] += (y_offset, x_offset)

    return out_bbox


def crop_bbox(
        bbox, y_slice=None, x_slice=None,
        allow_outside_center=True, return_param=False):
    """Translate bounding boxes to fit within the cropped area of an image.
    This method is mainly used together with image cropping.
    This method translates the coordinates of bounding boxes like
    :func:`~chainercv.transforms.translate_bbox`. In addition,
    this function truncates the bounding boxes to fit within the cropped area.
    If a bounding box does not overlap with the cropped area,
    this bounding box will be removed.
    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.
    Args:
        bbox (~numpy.ndarray): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        y_slice (slice): The slice of y axis.
        x_slice (slice): The slice of x axis.
        allow_outside_center (bool): If this argument is :obj:`False`,
            bounding boxes whose centers are outside of the cropped area
            are removed. The default value is :obj:`True`.
        return_param (bool): If :obj:`True`, this function returns
            indices of kept bounding boxes.
    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):
        If :obj:`return_param = False`, returns an array :obj:`bbox`.
        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`bbox, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.
        * **index** (*numpy.ndarray*): An array holding indices of used \
            bounding boxes.
    """

    t, b = _slice_to_bounds(y_slice)
    l, r = _slice_to_bounds(x_slice)
    crop_bb = np.array((t, l, b, r))

    if allow_outside_center:
        mask = np.ones(bbox.shape[0], dtype=bool)
    else:
        center = (bbox[:, :2] + bbox[:, 2:]) / 2
        mask = np.logical_and(crop_bb[:2] <= center, center < crop_bb[2:]) \
                 .all(axis=1)

    bbox = bbox.copy()
    bbox[:, :2] = np.maximum(bbox[:, :2], crop_bb[:2])
    bbox[:, 2:] = np.minimum(bbox[:, 2:], crop_bb[2:])
    bbox[:, :2] -= crop_bb[:2]
    bbox[:, 2:] -= crop_bb[:2]

    mask = np.logical_and(mask, (bbox[:, :2] < bbox[:, 2:]).all(axis=1))
    bbox = bbox[mask]

    if return_param:
        return bbox, {'index': np.flatnonzero(mask)}
    else:
        return bbox


def _slice_to_bounds(slice_):
    if slice_ is None:
        return 0, np.inf

    if slice_.start is None:
        l = 0
    else:
        l = slice_.start

    if slice_.stop is None:
        u = np.inf
    else:
        u = slice_.stop

    return l, u


def resize_bbox(bbox, in_size, out_size):
    """Resize bounding boxes according to image resize.
    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.
    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.
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


import random


def random_flip(img, y_random=False, x_random=False,
                return_param=False, copy=False):
    """Randomly flip an image in vertical or horizontal direction.
    Args:
        img (~numpy.ndarray): An array that gets flipped. This is in
            CHW format.
        y_random (bool): Randomly flip in vertical direction.
        x_random (bool): Randomly flip in horizontal direction.
        return_param (bool): Returns information of flip.
        copy (bool): If False, a view of :obj:`img` will be returned.
    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):
        If :obj:`return_param = False`,
        returns an array :obj:`out_img` that is the result of flipping.
        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`out_img, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.
        * **y_flip** (*bool*): Whether the image was flipped in the\
            vertical direction or not.
        * **x_flip** (*bool*): Whether the image was flipped in the\
            horizontal direction or not.
    """
    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        img = img[:, ::-1, :]
    if x_flip:
        img = img[:, :, ::-1]

    if copy:
        img = img.copy()

    if return_param:
        return img, {'y_flip': y_flip, 'x_flip': x_flip}
    else:
        return img


def flip_bbox(bbox, size, y_flip=False, x_flip=False):
    """Flip bounding boxes accordingly.
    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.
    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        y_flip (bool): Flip bounding box according to a vertical flip of
            an image.
        x_flip (bool): Flip bounding box according to a horizontal flip of
            an image.
    Returns:
        ~numpy.ndarray:
        Bounding boxes flipped according to the given flips.
    """
    H, W = size
    bbox = bbox.copy()
    if y_flip:
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max
    if x_flip:
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max
    return bbox