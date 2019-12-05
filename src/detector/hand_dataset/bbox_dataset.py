import copy
import logging

logger = logging.getLogger(__name__)

from chainer.dataset import DatasetMixin
import chainercv
import numpy as np

from detector.hand_dataset.common_dataset import palm2wrist
from detector.hand_dataset.geometry_utils import flip_point_zyx, rotate_point_zyx


def create_converter(hand_class):
    if hand_class == ["hand"]:
        class_converter = {
            "left": 0,
            "right": 0,
        }
        flip_converter = {0: 0}
    elif hand_class == ["left", "right"]:
        class_converter = {
            "left": 0,
            "right": 1,
        }
        flip_converter = {
            0: 1,
            1: 0,
        }
    else:
        raise ValueError('should be ["hand"] or["left","right"]')
    return class_converter, flip_converter


def scale(image, camera, crop2d):
    C, H, W = image.shape
    length = min(H, W)
    size = min(crop2d)
    s = size / length
    image = chainercv.transforms.scale(image, size=size, fit_short=True)
    camera = camera.scale_camera(y_scale=s, x_scale=s)
    return image, camera


def flip_hand(image, joint_zyx, camera, x_flip=False, y_flip=False):
    C, H, W = image.shape
    joint_zyx_flipped = np.empty(joint_zyx.shape)
    for i in range(len(joint_zyx)):
        joint_zyx_flipped[i] = flip_point_zyx(camera, joint_zyx[i], (H, W), y_flip=y_flip, x_flip=x_flip)

    image_flipped = chainercv.transforms.flip(image, y_flip=y_flip, x_flip=x_flip)
    return image_flipped, joint_zyx_flipped


def rotate_hand(image, joint, camera, angle):
    C, H, W = image.shape
    center_vu = (H / 2, W / 2)
    image_angle = angle
    # to make compatibility between chainercv.transforms.rotate and rot_point_vu
    point_angle = -angle
    joint_rot = np.empty(joint.shape)
    for i in range(len(joint)):
        joint_rot[i] = rotate_point_zyx(camera, joint[i], point_angle, center_vu)
    image_rot = chainercv.transforms.rotate(image, image_angle, expand=False)
    return image_rot, joint_rot


def preprocess(image, joint, camera, hand_side, param, flip_converter):
    # random rotate
    image, joint = rotate_hand(image, joint, camera, param["angle"])
    # scale image
    image, camera = scale(image, camera, param["crop2d"])

    image, joint = flip_hand(
        image, joint, camera,
        y_flip=param["y_flip"],
        x_flip=param["x_flip"],
    )

    # random flip
    classes = []
    for i in range(len(hand_side)):
        if param["y_flip"]:
            hand_side[i] = flip_converter[hand_side[i]]
        if param["x_flip"]:
            hand_side[i] = flip_converter[hand_side[i]]
        classes.append(hand_side[i])

    # define bbox
    bboxes = []
    for i in range(len(joint)):
        zyx = palm2wrist(joint[i])
        vu = camera.zyx2vu(zyx)
        vmax = np.max(vu[:, 0])
        vmin = np.min(vu[:, 0])
        umax = np.max(vu[:, 1])
        umin = np.min(vu[:, 1])
        vlen, ulen = vmax - vmin, umax - umin
        vc, uc = (vmax + vmin) / 2, (umax + umin) / 2
        s = 1.25
        length = s * max(vlen, ulen)
        bbox = [
            vc - length / 2,
            uc - length / 2,
            vc + length / 2,
            uc + length / 2,
        ]
        bboxes.append(bbox)
    bboxes = np.array(bboxes, dtype=np.float32)
    classes = np.array(classes, dtype=np.int32)
    return image, bboxes, classes


class HandBBoxDataset(DatasetMixin):
    def __init__(self, base, param):
        self.base = base
        self.do_augmentation = (self.base.mode == "train")
        self.rgb_camera = base.rgb_camera
        self.enable_x_flip = param["enable_x_flip"]
        self.enable_y_flip = param["enable_y_flip"]
        self.angle_range = param["angle_range"]
        self.hand_class = param["hand_class"]
        class_converter, flip_converter = create_converter(self.hand_class)
        self.class_converter = class_converter
        self.flip_converter = flip_converter

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        base_example = copy.deepcopy(self.base.get_example(i))

        # initialize parameter for preprocessing
        y_flip = x_flip = False
        angle = 0
        if self.do_augmentation:
            if self.enable_x_flip:
                x_flip = np.random.choice([True, False])
            if self.enable_y_flip:
                y_flip = np.random.choice([True, False])
            angle = np.random.choice(self.angle_range)

        param = {
            "crop2d": (300, 300),
            "x_flip": x_flip,
            "y_flip": y_flip,
            "angle": angle,
        }

        rgb_joint_zyx = base_example["rgb_joint"]
        hand_side = base_example["hand_side"]
        if isinstance(hand_side, str):
            # single hand
            hand_side = np.array([self.class_converter[hand_side]])
        else:
            # multiple hand
            hand_side = np.asarray([self.class_converter[h] for h in hand_side])

        if rgb_joint_zyx.ndim == 2:
            # single hand
            rgb_joint_zyx = np.expand_dims(rgb_joint_zyx, axis=0)

        rgb = chainercv.utils.read_image(base_example["rgb_path"])
        camera = base_example.get("rgb_camera", self.rgb_camera)

        rgb_img, rgb_bbox, rgb_class = preprocess(rgb, rgb_joint_zyx, camera, hand_side, param, self.flip_converter)

        return rgb_img, rgb_bbox, rgb_class
