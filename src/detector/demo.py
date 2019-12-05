import argparse
import configparser
import logging

logger = logging.getLogger(__name__)
import os

import chainer
import chainercv
from PIL import ImageDraw, Image

import cv2
import numpy as np

from detector.hand_dataset.image_utils import COLOR_MAP
import utils


def draw_hands(pil_image, bbox, label, score, hand_class):
    drawer = ImageDraw.Draw(pil_image)
    for b, l, s in zip(bbox, label, score):
        ymin, xmin, ymax, xmax = b.astype(int)
        name = hand_class[l]
        color = COLOR_MAP[name]
        drawer.rectangle(
            xy=[xmin, ymin, xmax, ymax],
            fill=None,
            outline=color
        )
    return pil_image


def main(args):
    logger.info("> setup config")
    trained = args.trained
    config = configparser.ConfigParser()
    config.read(os.path.join(trained, "detector", "config.ini"))
    model_param = utils.get_config(config)
    logger.info("> setup model")
    model = utils.create_ssd_model(model_param)
    chainer.serializers.load_npz(
        os.path.join(trained, "detector", "bestmodel.npz"),
        model
    )
    hand_class = config["model_param"]["hand_class"].split(",")
    hand_class = [k.strip() for k in hand_class]
    logger.info("> hand_class = {}".format(hand_class))
    logger.info("> set up camera, cameraId = {}".format(args.camera))
    cap = cv2.VideoCapture(args.camera)
    if cap.isOpened() is False:
        raise Exception("Error opening video stream of file")
    logger.info("> finish setup")
    logger.info("> start demo")
    inH, inW = model_param["input_size"], model_param["input_size"]

    while cap.isOpened():
        ret_val, image = cap.read()
        # convert color BGR -> RGB and HWC -> CHW
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        _, cH, cW = image.shape
        sz = min(cH, cW)
        image = chainercv.transforms.center_crop(image, (sz, sz))
        #image = chainercv.transforms.resize(image, (inH, inW))
        with chainer.using_config('train', False):
            bboxes, labels, scores = model.predict(np.expand_dims(image, axis=0).astype(np.float32))
        bbox, label, score = bboxes[0], labels[0], scores[0]
        logger.info("{} {} {}".format(bbox, label, score))
        # CHW -> HWC
        image = image.transpose(1, 2, 0)
        pil_image = Image.fromarray(image)
        pil_image = draw_hands(pil_image, bbox, label, score, hand_class)
        cv2.namedWindow("HandDetectorDemo", cv2.WINDOW_AUTOSIZE)
        image = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
        cv2.imshow("HandDetectorDemo", image)
        if cv2.waitKey(1) == 27:  # press ESC to stop
            break
        cv2.waitKey(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("trained", type=str, default="trained")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--outsize", type=int, default=600)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
