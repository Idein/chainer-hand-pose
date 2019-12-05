import configparser
import copy
import logging

logger = logging.getLogger(__name__)

from chainer.dataset import DatasetMixin
import chainercv
from chainercv.links.model.ssd import random_distort
import numpy as np

from pose.hand_dataset.geometry_utils import crop_domain2d, crop_domain3d
from pose.hand_dataset.geometry_utils import flip_point_zyx, rotate_point_zyx


def crop_handwith2d(image, joint, camera, aug_param):
    vu = camera.zyx2vu(joint)
    vmax = np.max(vu[:, 0])
    vmin = np.min(vu[:, 0])
    umax = np.max(vu[:, 1])
    umin = np.min(vu[:, 1])
    vlen, ulen = vmax - vmin, umax - umin
    vc, uc = (vmax + vmin) / 2, (umax + umin) / 2
    if aug_param.get("do_oscillate", False):
        scale = np.random.choice(aug_param["scale_range"])
        vscale = np.random.choice(aug_param["shift_range"])
        uscale = np.random.choice(aug_param["shift_range"])
        length = scale * max(65, vlen, ulen)
        vshift = vscale * length
        ushift = uscale * length
    else:
        scale = 1.25
        length = scale * max(65, vlen, ulen)
        vshift = ushift = 0

    domain = [
        vshift + vc - length / 2,
        ushift + uc - length / 2,
        vshift + vc + length / 2,
        ushift + uc + length / 2,
    ]
    domain = list(map(int, domain))
    image_cropped, crop_param = crop_domain2d(image, domain)
    camera_cropped = camera.translate_camera(
        y_offset=crop_param["y_offset"],
        x_offset=crop_param["x_offset"]
    )
    return image_cropped, camera_cropped,domain


def crop_handwith3d(image, joint, camera, domain3d, aug_param):
    image_cropped, camera_cropped = crop_domain3d(
        image, joint, camera,
        domain3d,
        aug_param,
    )
    return image_cropped, camera_cropped


def scale(image, camera, crop2d):
    C, H, W = image.shape
    length = min(H, W)
    size = min(crop2d)
    scale = size / length
    image = chainercv.transforms.scale(image, size=size, fit_short=True)
    camera = camera.scale_camera(y_scale=scale, x_scale=scale)
    return image, camera


def flip_hand(image, joint_zyx, camera, x_flip=False, y_flip=False):
    C, H, W = image.shape
    joint_zyx_flipped = flip_point_zyx(
        camera, joint_zyx, (H, W), y_flip=y_flip, x_flip=x_flip)
    image_flipped = chainercv.transforms.flip(image, y_flip=y_flip, x_flip=x_flip)
    return image_flipped, joint_zyx_flipped


def rotate_hand(image, joint, camera, angle):
    C, H, W = image.shape
    center_vu = (H / 2, W / 2)
    image_angle = angle
    # to make compatibility between chainercv.transforms.rotate and rot_point_vu
    point_angle = -angle
    joint_rot = rotate_point_zyx(camera, joint, point_angle, center_vu)
    image_rot = chainercv.transforms.rotate(image, image_angle, expand=False)
    return image_rot, joint_rot


def resize_contain(image, camera, size, fill=0):
    _, inH, inW = image.shape
    image_resized, resize_param = chainercv.transforms.resize_contain(
        image,
        size=size,
        return_param=True,
        fill=fill,
    )
    y_scale, x_scale = resize_param["scaled_size"] / np.array([inH, inW])

    camera_scaled = camera.scale_camera(y_scale=y_scale, x_scale=x_scale)
    camera_resized = camera_scaled.translate_camera(
        y_offset=resize_param["y_offset"],
        x_offset=resize_param["x_offset"],
    )
    return image_resized, camera_resized


def preprocess(image, joint, camera, param):
    logger.debug("rotate")
    # random rotate
    image, joint = rotate_hand(image, joint, camera, angle=param["angle"])
    if param["add_noise"]:
        noise = np.random.normal(0, 1, joint.shape)
        joint = joint + noise
    # crop
    logger.debug("crop")
    image, camera, domain = crop_handwith2d(
        image,
        joint,
        camera,
        param["oscillation"],
    )

    logger.debug("scale")
    image, camera = scale(image, camera, param["crop2d"])

    logger.debug("flip")
    # random flip
    image, joint = flip_hand(
        image, joint, camera,
        y_flip=param["y_flip"],
        x_flip=param["x_flip"],
    )
    logger.debug("flip")
    # scale image so that target image contains a specific domain
    image, camera = resize_contain(image, camera, size=param["crop2d"], fill=0)
    if param["do_color_augmentation"]:
        logger.debug("distort image")
        image = random_distort(image)
    return image, joint, camera, domain


class HandPoseDataset(DatasetMixin):
    def __init__(self, base, param):
        self.base = base
        self.do_augmentation = (self.base.mode == "train")
        self.rgb_camera = base.rgb_camera
        self.depth_camera = base.depth_camera
        self.n_joints = base.n_joints
        self.use_rgb = param["use_rgb"]
        self.use_depth = param["use_depth"]
        self.crop3d = param["cube"]  # DHW
        self.crop2d = param["imsize"]
        self.z_size = param["cube"][0]
        self.enable_x_flip = param["enable_x_flip"]
        self.enable_y_flip = param["enable_y_flip"]
        self.angle_range = param["angle_range"]
        self.oscillation = param["oscillation"]

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        base_example = copy.deepcopy(self.base.get_example(i))

        # initialize parameter for preprocessing
        y_flip = x_flip = False
        angle = 0
        oscillation = copy.deepcopy(self.oscillation)
        if self.do_augmentation:
            if self.enable_x_flip:
                x_flip = np.random.choice([True, False])
            if self.enable_y_flip:
                y_flip = np.random.choice([True, False])
            angle = np.random.choice(self.angle_range)
        else:
            oscillation["do_oscillate"] = False
        if base_example.get("hand_side") == "left":
            # force set
            x_flip = True

        param = {
            "do_color_augmentation": self.do_augmentation,
            "add_noise": self.do_augmentation,
            "oscillation": self.oscillation,
            "crop3d": self.crop3d,
            "z_size": self.z_size,
            "crop2d": self.crop2d,
            "x_flip": x_flip,
            "y_flip": y_flip,
            "angle": angle,
        }

        example = {
            "param": param,
        }

        if self.use_rgb:
            rgb_joint_zyx = base_example["rgb_joint"]
            logger.debug("load rgb")
            logger.debug("read image")
            rgb = chainercv.utils.read_image(base_example["rgb_path"])
            logger.debug("process image joint and camera parameters")
            camera = base_example.get("rgb_camera", self.rgb_camera)
            rgb_img, rgb_joint, rgb_camera,domain = preprocess(rgb, rgb_joint_zyx, camera, param)
            example["rgb"] = rgb_img
            example["rgb_joint"] = rgb_joint
            example["rgb_camera"] = rgb_camera
            example["domain"]=domain
        if self.use_depth:
            depth_joint_zyx = base_example["depth_joint"]
            logger.debug("load depth")
            param["do_color_augmentation"] = False
            depth = self.base.read_depth(base_example["depth_path"])
            camera = base_example.get("depth_camera", self.depth_camera)
            depth_img, depth_joint, depth_camera,domain = preprocess(depth, depth_joint_zyx, camera, param)
            example["depth"] = depth_img
            example["depth_joint"] = depth_joint
            example["depth_camera"] = depth_camera
            example["domain"] = domain
        return example
