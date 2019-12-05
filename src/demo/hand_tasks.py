import logging

logger = logging.getLogger(__name__)

import traceback

from PIL import Image, ImageOps
import numpy as np
from actfw.task import Pipe

from hand_pose_utils import normalize_rgb
from transforms import flip_point


class DetectorTask(Pipe):
    def __init__(self, detector, hand_class, capH, capW):
        super(DetectorTask, self).__init__()
        self.detector = detector
        self.insize = self.detector.insize
        self.mean = self.detector.mean
        self.hand_class = hand_class
        self.capH = capH
        self.capW = capW

    def _prepare(self, pil_img):
        """
        preprocess image and return array for inference
        1. convert pil_img to numpy array
        2. preprocess the array and transpose axis HWC -> CHW.
        """
        resized = pil_img.resize((self.insize, self.insize))
        ret = np.asarray(resized, dtype=np.float32)
        ret -= self.mean
        return np.asarray(ret.transpose(2, 0, 1), order="C")

    def proc(self, frame):
        try:
            # assume the layout of the img is HWC
            pil_img = Image.frombuffer(
                'RGB',
                (self.capW, self.capH),
                frame.getvalue(),
                'raw',
                'RGB'
            )
            # center crop
            W, H = pil_img.size
            sz = min(H, W)
            left = (W - sz) // 2
            right = left + sz
            upper = (H - sz) // 2
            lower = upper + sz
            pil_img = pil_img.crop((left, upper, right, lower))
            sizeW, sizeH = pil_img.size
            bbox, label, score = self.detector.predict(self._prepare(pil_img), sizeH, sizeW)
            do_flip = []
            class_name = [self.hand_class[l] for l in label]
            do_flip = [self.hand_class[l] == "left" for l in label]
            logger.info("bbox={} label={} score={}".format(bbox, label, score))
            pipe = {
                "pil_img": pil_img,
                "bbox": bbox,
                "label": label,
                "class_name": class_name,
                "do_flip": do_flip,
                "score": score,
            }
            return pipe
        except Exception as e:
            logger.error("error in {} with error \n{}".format(self.__class__.__name__, e))
            traceback.print_exc()
            # notify stop event to parent application
            self.app.running = False
            return None


class PoseTask(Pipe):
    def __init__(self, pose, hand_class):
        super(PoseTask, self).__init__()
        self.pose = pose
        self.inC = self.pose.inC
        self.inH = self.pose.inH
        self.inW = self.pose.inW
        self.outH = self.pose.outH
        self.outW = self.pose.outW
        self.hand_class = hand_class

    def _prepare(self, pil_img, x_flip):
        # preprocess image for inference
        # 1. check flip 2. resize and 3. normalize
        # output array should have layout of CHW
        if x_flip:
            # flip x axis
            pil_img = ImageOps.mirror(pil_img)
        pil_img = pil_img.resize((self.inW, self.inH))
        ret = np.asarray(pil_img)
        ret = normalize_rgb(ret)
        ret = np.asarray(ret.transpose(2, 0, 1), dtype=np.float32, order="C")
        return ret

    def proc(self, pipe):
        try:
            # TODO use pil_img.crop
            pil_img = pipe["pil_img"]
            bbox = pipe["bbox"]
            label = pipe["label"]
            do_flip = pipe["do_flip"]
            oriW, oriH = pil_img.size
            pred_vu = []
            found = {
                "left": False,
                "right": False,
                "hand": False,
            }
            patch_x = []
            patch_size = []
            patch_offset = []
            is_detected = False
            for b, x_flip, l in zip(bbox, do_flip, label):
                is_detected = True
                # extract patch
                ymin, xmin, ymax, xmax = b
                ulen = xmax - xmin
                vlen = ymax - ymin
                boxscale = 1.
                boxlen = int(boxscale * max(ulen, vlen))
                uc = int((xmax + xmin) / 2)
                vc = int((ymax + ymin) / 2)
                umin = max(0, uc - boxlen // 2)
                umax = min(oriW, uc + boxlen // 2)
                vmin = max(0, vc - boxlen // 2)
                vmax = min(oriH, vc + boxlen // 2)
                patch = pil_img.crop((umin, vmin, umax, vmax))
                patchW, patchH = patch.size

                name = self.hand_class[l]
                if self.hand_class == ["left", "right"]:
                    if found[name]:
                        continue
                    else:
                        patch_x.append(self._prepare(patch, x_flip))
                        patch_size.append((patchH, patchW))
                        patch_offset.append((vmin, umin))
                        found[name] = True
                    if found["left"] and found["right"]:
                        # already gather for each hands
                        break
                else:
                    patch_x.append(self._prepare(patch, x_flip))
                    patch_size.append((patchH, patchW))
                    patch_offset.append((vmin, umin))
                    # use only one hand
                    break

            kp_vu, kp_zyx = [], []
            if is_detected:

                # if False:
                #    # forward with patch_x more than one batch size
                #    patch_x = np.asarray(patch_x, dtype=np.float32, order="C")
                #    kp_vu, kp_zyx = self.pose.predict(patch_x)
                if True:
                    # forward with single x
                    for x in patch_x:
                        vu, zyx = self.pose.predict(x)
                        vu, zyx = vu[0], zyx[0]
                        kp_vu.append(vu)
                        kp_zyx.append(zyx)

            iterble = [kp_vu, patch_offset, patch_size, do_flip]
            for vu, offset, (patchH, patchW), x_flip in zip(*iterble):
                if x_flip:
                    vu = flip_point(
                        vu[np.newaxis],
                        (self.inH, self.inW),
                        x_flip=True,
                    ).squeeze(axis=0)

                vu = np.array([[patchH / self.inH, patchW / self.inW]]) * vu + np.array([offset])
                pred_vu.append(vu)

            pipe["bbox"] = pipe["bbox"][:len(pred_vu)]
            pipe["label"] = pipe["label"][:len(pred_vu)]
            pipe["pred_vu"] = pred_vu
            pipe["pred_zyx"] = kp_zyx

        except Exception as e:
            logger.error("error in {} with error \n{}".format(self.__class__.__name__, e))
            traceback.print_exc()
            # notify stop event to parent application
            self.app.running = False

        return pipe
