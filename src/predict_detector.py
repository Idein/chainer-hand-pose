import argparse
import configparser
import glob
from importlib import import_module
import logging
logger = logging.getLogger(__name__)

import os

import matplotlib.pyplot as plt
import chainer
import chainercv
from chainercv.visualizations import vis_bbox
from chainer import serializers

import detector.utils as detector_utils


def save_images(image_path, model, out_root, label_names):
    image_path_list = [x for x in glob.glob(os.path.join(image_path, '*.jpg'))]
    image_path_list += [x for x in glob.glob(os.path.join(image_path, '*.png'))]
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    fig, ax = plt.subplots()
    for _img in image_path_list:
        img = chainercv.utils.read_image(_img, color=True)
        with chainer.using_config('train', False):
            bboxes, labels, scores = model.predict([img])
        bbox, label, score = bboxes[0], labels[0], scores[0]
        logger.info("{} {} {}".format(bbox, label, score))
        vis_bbox(img, bbox, label, score, label_names=label_names, ax=ax)
        f_name = os.path.join(out_root, os.path.basename(_img))
        plt.savefig(f_name)
        ax.clear()


def predict(model_path, img_path, out_path, config):
    train_param = detector_utils.get_config(config)
    model = detector_utils.create_ssd_model(train_param)
    serializers.load_npz(model_path, model)
    hand_class = config["model_param"]["hand_class"].split(",")
    hand_class = [k.strip() for k in hand_class]
    save_images(img_path, model, out_path, hand_class)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str)
    parser.add_argument('--config_path', type=str)
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--out_path', type=str)

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path)

    predict(args.model_path, args.img_path, args.out_path, config)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
