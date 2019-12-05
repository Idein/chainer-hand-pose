import logging

logger = logging.getLogger(__name__)
import traceback
import time

import actfw
from actfw.task import Consumer
from PIL import Image
import numpy as np

from drawer import draw_axis, draw_bbox, draw_hands
# detector
from hand_detector_utils import COLOR_MAP as DETECTOR_COLOR_MAP
# pose
from hand_pose_utils import format_kp_proj
from hand_pose_utils import COLOR_MAP as POSE_COLOR_MAP
from config import CAPTURE_WIDTH, CAPTURE_HEIGHT
from config import DISPLAY_WIDTH, DISPLAY_HEIGHT
from pil_utils import concat_h, paste_contain


class Presenter(Consumer):
    def __init__(self, hand_class, edges):
        super(Presenter, self).__init__()
        self.hand_class = hand_class
        self.edges = edges
        self.canvas = Image.new("RGB", (CAPTURE_WIDTH, CAPTURE_HEIGHT), (0, 0, 0))

    def get_proj(self, kp_zyx, label, projboxH, projboxW):
        proj_pts = []
        for i, (zyx, l) in enumerate(zip(kp_zyx, label)):
            name = self.hand_class[l]
            kp_proj = [zyx[:, [1, 2]], zyx[:, [1, 0]], zyx[:, [0, 2]]]
            if self.hand_class == ["left", "right"]:
                yx = format_kp_proj(
                    kp_proj[0],
                    projboxH, projboxW,
                    0 * projboxH, 0 if name == "right" else projboxW,
                    x_flip=True if name == "left" else False
                )
                yz = format_kp_proj(
                    kp_proj[1], projboxH, projboxW,
                    1 * projboxH, 0 if name == "right" else projboxW,
                    x_flip=False if name == "left" else True
                )
                zx = format_kp_proj(
                    kp_proj[2], projboxH, projboxW,
                    2 * projboxH, 0 if name == "right" else projboxW,
                    x_flip=True if name == "left" else False,
                    y_flip=True
                )
            else:
                yx = format_kp_proj(
                    kp_proj[0],
                    projboxH, projboxW,
                    0 * projboxH, i * projboxW,
                    x_flip=False,
                )
                yz = format_kp_proj(
                    kp_proj[1], projboxH, projboxW,
                    1 * projboxH, i * projboxW,
                    x_flip=True,
                )
                zx = format_kp_proj(
                    kp_proj[2], projboxH, projboxW,
                    2 * projboxH, i * projboxW,
                    x_flip=False,
                    y_flip=True
                )
            proj_pts += [yx, yz, zx]
        return proj_pts

    def get_image(self, pipe):
        try:
            time_draw = time.time()
            pil_img = pipe["pil_img"]
            oriW, oriH = pil_img.size
            projH, projW = oriH, len(self.hand_class) * oriW // 3
            projboxH = projH // 3
            projboxW = projW // len(self.hand_class)
            bbox = pipe["bbox"]
            label = pipe["label"]
            pred_vu = pipe["pred_vu"]
            pred_zyx = pipe["pred_zyx"]
            proj_pts = self.get_proj(pred_zyx, label, projboxH, projboxW)
            pil_proj = Image.new("RGB", (projW, projH), (0, 0, 0))
            draw_bbox(pil_img, bbox, label, self.hand_class, DETECTOR_COLOR_MAP)
            for i_w in range(len(self.hand_class)):
                iterble = (
                    range(3),
                    ["green", "green", "blue"], ["red", "blue", "red"],
                    ["Y", "Y", "Z"], ["X", "Z", "X"]
                )
                for i_h, h_color, w_color, h_text, w_text in zip(*iterble):
                    y_offset = i_h * projboxH
                    x_offset = i_w * projboxW
                    axisH = projboxH
                    axisW = projboxW
                    draw_axis(pil_proj, y_offset, x_offset, axisH, axisW, h_color, w_color, h_text, w_text)

            draw_hands(pil_img, pred_vu, self.edges, POSE_COLOR_MAP)
            proj_pts = proj_pts[:3 * len(self.hand_class)]  # proj hand domain with shape 3x2
            draw_hands(pil_proj, proj_pts, self.edges, POSE_COLOR_MAP)
            logger.info("> draw result {} [msec]".format(1000 * (time.time() - time_draw)))
            image_show = paste_contain(self.canvas, concat_h(pil_img, pil_proj))
            return image_show
        except Exception as e:
            logger.error("{}".format(e))
            traceback.print_exc()
            self.app.running = False


class DesktopPresenter(Presenter):

    def __init__(self, hand_class, edges):
        super(DesktopPresenter, self).__init__(hand_class, edges)
        self.hand_class = hand_class
        self.edges = edges
        self.img = None

    def proc(self, pipe):
        self.img = np.asarray(self.get_image(pipe))


class PiCameraPresenter(Presenter):

    def __init__(self, settings, camera, cmd, hand_class, edges):
        super(PiCameraPresenter, self).__init__(hand_class, edges)
        self.settings = settings
        self.camera = camera
        self.cmd = cmd
        if self.settings["display"]:
            self.display = actfw.Display(camera, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    def proc(self, pipe):
        image_show = self.get_image(pipe)
        image_show = image_show.resize((CAPTURE_WIDTH, CAPTURE_HEIGHT))
        self.cmd.update_image(image_show)
        actfw.heartbeat()
        if self.settings['display']:
            self.display.update(
                (0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT),
                image_show.tobytes(),
                (CAPTURE_WIDTH, CAPTURE_HEIGHT),
                'rgb'
            )
