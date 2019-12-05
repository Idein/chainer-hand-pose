import argparse
import logging

logger = logging.getLogger(__name__)

import signal
import time
import traceback

import actfw
from actfw.task import Pipe, Task, Consumer
import cv2
import numpy as np
from PIL import Image

from actfw_opencv import OpenCVCamera, CenterCropScale, OpenCVApplication
# detector
from hand_detector import select_detector
# pose
from hand_pose import select_pose
# task
from hand_tasks import DetectorTask, PoseTask
from act_utils import Sequential
# presenter
from presenter import DesktopPresenter
from config import CAPTURE_WIDTH, CAPTURE_HEIGHT


def main_desktop(args):
    # desktop users only

    capture = cv2.VideoCapture(0)
    logger.info("CAP_PROP_FPS {}".format(capture.get(cv2.CAP_PROP_FPS)))
    # testout your camera works
    ret_val, img = capture.read()
    if not ret_val:
        raise Exception("OpenCV Camera Error")

    file_type = args.file_type
    if file_type == "npz":
        # left right class
        detector_path = "../../result/release"
        # one class
        # detector_path = "../../result/release_oneclass"
        pose_path = "../../result/release"
    elif file_type == "nnoir":
        detector_path = "./"
        pose_path = "./"
    else:
        ValueError("invalid file_type {}".format(file_type))

    preprocessor = CenterCropScale(inH=CAPTURE_HEIGHT, inW=CAPTURE_WIDTH, color="RGB")
    detector = select_detector(detector_path, file_type)
    pose = select_pose(pose_path, file_type)
    hand_class = detector.param["hand_class"]
    # setup Task: Producer,Consumer or Pipe etc...
    cam = OpenCVCamera(preprocessor, capture)
    dt = DetectorTask(detector, hand_class, capH=CAPTURE_HEIGHT, capW=CAPTURE_WIDTH)
    pt = PoseTask(pose, hand_class)
    presenter = DesktopPresenter(hand_class, pose.param["edges"])
    seq = Sequential([dt, pt])
    # connect tasks
    cam.connect(seq)
    seq.connect(presenter)
    # setup application
    app = OpenCVApplication(capture_color=cam.color)
    app.register_task(cam)
    app.register_task(seq)
    seq.register_app(app)
    app.register_task(presenter)
    # run
    app.run()


def parse_argument():
    parser = argparse.ArgumentParser(description="Hand Pose Estimation")
    parser.add_argument("--file_type", type=str, choices=["npz", "nnoir"], default="npz")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # desktop mode
    logging.basicConfig(level=logging.INFO)
    args = parse_argument()
    main_desktop(args)
