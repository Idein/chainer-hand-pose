import argparse
import configparser
import logging

logger = logging.getLogger(__name__)
import os
import time

import chainercv
import cv2
import numpy as np
from PIL import ImageDraw, Image

import chainer

# detector
import detector.utils as detector_utils
from detector.hand_dataset.image_utils import COLOR_MAP as DETECTOR_COLOR_MAP
# pose
from pose.hand_dataset.selector import select_dataset as select_pose_dataset
from pose.models.selector import select_model as select_pose_model
from pose.hand_dataset.image_utils import normalize_rgb
from pose.hand_dataset.common_dataset import COLOR_MAP as POSE_COLOR_MAP


def to_device(model):
    logger.info('> cuda enable {}'.format(chainer.backends.cuda.available))
    logger.info('> ideep enable {}'.format(chainer.backends.intel64.is_ideep_available()))
    if chainer.backends.cuda.available:
        logger.info('> use GPU mode')
        model.to_gpu()
    elif chainer.backends.intel64.is_ideep_available():
        logger.info('> Intel64 mode')
        model.to_intel64()
    return model


def format_kp_proj(point, outH, outW, offsetH=0, offsetW=0, x_flip=False, y_flip=False):
    vmin = np.min(point[:, 0])
    umin = np.min(point[:, 1])
    vmax = np.max(point[:, 0])
    umax = np.max(point[:, 1])
    ulen = vmax - vmin
    vlen = umax - umin
    scale = min(outH, outW) / max(ulen, vlen)
    offset = np.array([vmin, umin])
    point = scale * (point - offset)
    point = chainercv.transforms.flip_point(
        point[np.newaxis],
        (outH, outW),
        x_flip=x_flip,
        y_flip=y_flip,
    ).squeeze(axis=0)
    point = point + np.array([offsetH, offsetW])
    return point


def draw_hands(pil_image, kp_vu, edges):
    drawer = ImageDraw.Draw(pil_image)
    r = 2
    for p in kp_vu:
        # yx yx ... -> xy xy ...
        for i, (x, y) in enumerate(p[:, ::-1]):
            color = tuple(POSE_COLOR_MAP[i])
            drawer.ellipse((x - r, y - r, x + r, y + r),
                           fill=color)
        for s, t in edges:
            sy, sx = p[s].astype(int)
            ty, tx = p[t].astype(int)
            color = tuple(POSE_COLOR_MAP[s, t])
            drawer.line([(sx, sy), (tx, ty)], fill=color)


def draw_bbox(pil_image, bbox, label, hand_class, color_map=None):
    drawer = ImageDraw.Draw(pil_image)
    for b, l in zip(bbox, label):
        ymin, xmin, ymax, xmax = b.astype(int)
        name = hand_class[l]
        if color_map is None:
            color = (128, 128, 128)
        else:
            color = color_map[name]
        drawer.rectangle(
            xy=[xmin, ymin, xmax, ymax],
            fill=None,
            outline=color
        )


def get_result_ppn(image, model, pose_param):
    time_ppn = time.time()
    with chainer.using_config("train", False):
        ret = model.predict(
            model.xp.expand_dims(
                normalize_rgb(model.xp.asarray(image)).astype(model.xp.float32),
                axis=0
            )
        )
    logger.info("> ppn inference {} [msec]".format(1000 * (time.time() - time_ppn)))
    time_postprocessing = time.time()
    resp, conf, x, y, w, h, v = ret
    resp = chainer.backends.cuda.to_cpu(resp.array)
    conf = chainer.backends.cuda.to_cpu(conf.array)
    # w = chainer.backends.cuda.to_cpu(w.array)
    # h = chainer.backends.cuda.to_cpu(h.array)
    x = chainer.backends.cuda.to_cpu(x.array)
    y = chainer.backends.cuda.to_cpu(y.array)
    v = chainer.backends.cuda.to_cpu(v.array)
    resp = np.squeeze(resp, axis=0)
    conf = np.squeeze(conf, axis=0)
    x = np.squeeze(x, axis=0)
    y = np.squeeze(y, axis=0)
    # w = np.squeeze(w, axis=0)
    # h = np.squeeze(h, axis=0)
    v = np.squeeze(v, axis=0)
    color_map = pose_param["color_map"]
    keypoint_names = pose_param["keypoint_names"]
    edges = pose_param["edges"]
    delta = resp * conf
    scaleH = pose_param["inH"] / model.outsize[0]
    scaleW = pose_param["inW"] / model.outsize[1]
    joint2d = {}
    grid_position = {}

    for kname in keypoint_names:
        if "mcp" in kname or "root" == kname:
            i = keypoint_names.index(kname)
            u_ind = np.unravel_index(np.argmax(delta[i]), delta[i].shape)
            y_offset, x_offset = u_ind
            joint2d[kname] = [
                scaleH * (y_offset + y[i][u_ind]),
                scaleW * (x_offset + x[i][u_ind])
            ]
            grid_position[kname] = u_ind

    for f in ["thumb", "index", "middle", "ring", "little"]:
        for p, q in zip(["mcp", "pip", "dip"], ["pip", "dip", "tip"]):
            f_p = "_".join([f, p])
            f_q = "_".join([f, q])
            p_h, p_w = grid_position[f_p]
            i = keypoint_names.index(f_q)
            sz = 1 if q == "tip" else 2
            hslice = slice(max(0, p_h - sz), min(model.outsize[0], p_h + sz + 1))
            wslice = slice(max(0, p_w - sz), min(model.outsize[1], p_w + sz + 1))
            target = delta[i][hslice, wslice]
            q_h, q_w = np.unravel_index(np.argmax(target), target.shape)
            y_offset = (p_h - sz) + q_h if p_h - sz >= 0 else q_h
            x_offset = (p_w - sz) + q_w if p_w - sz >= 0 else q_w
            joint2d[f_q] = [
                scaleH * (y_offset + y[i][(y_offset, x_offset)]),
                scaleW * (x_offset + x[i][(y_offset, x_offset)])
            ]
            grid_position[f_q] = (y_offset, x_offset)
    kp_vu = np.array([joint2d[k] for k in keypoint_names])

    kp_zyx = np.zeros((len(keypoint_names), 3))
    for ei, (s, t) in enumerate(edges):
        u_ind = grid_position[keypoint_names[s]]
        orien = v[ei, :, u_ind[0], u_ind[1]]
        orien = orien / np.linalg.norm(orien)
        # elen = np.sqrt(np.sum(np.square(gt_3dj[t] - gt_3dj[s])))
        elen = 1.5 if s == 0 else 1
        kp_zyx[t] = kp_zyx[s] + orien * elen
    logger.info("> ppn_postprocessing {} [msec]".format(1000 * (time.time() - time_postprocessing)))
    return kp_vu, kp_zyx


def setup_detector(args):
    logger.info("> setup config")
    detector_path = args.detector
    config = configparser.ConfigParser()
    config.read(os.path.join(detector_path, "detector", "config.ini"))
    detector_param = detector_utils.get_config(config)
    logger.info("> setup model")
    model = detector_utils.create_ssd_model(detector_param)
    chainer.serializers.load_npz(
        os.path.join(detector_path, "detector", "bestmodel.npz"),
        model
    )
    hand_class = config["model_param"]["hand_class"].split(",")
    hand_class = [k.strip() for k in hand_class]

    detector_param = {
        "hand_class": hand_class,
        "inH": config.getint("model_param", "input_size"),
        "inW": config.getint("model_param", "input_size"),
    }
    model = to_device(model)
    return model, detector_param


def setup_pose(args):
    pose_path = args.pose
    config = configparser.ConfigParser()
    path = os.path.expanduser(os.path.join(pose_path, "pose", "config.ini"))
    logger.info("read {}".format(path))
    config.read(path, 'UTF-8')

    logger.info("setup devices")

    pose_param = select_pose_dataset(config, return_data=["hand_param"])
    model_path = os.path.expanduser(os.path.join(pose_path, "pose", "bestmodel.npz"))

    logger.info("> restore model")
    model = select_pose_model(config, pose_param)
    logger.info("> model.device = {}".format(model.device))

    logger.info("> restore models")
    chainer.serializers.load_npz(model_path, model)
    model = to_device(model)
    return model, pose_param


def main(args):
    logger.info("> setup detector")
    detector_model, detector_param = setup_detector(args)
    hand_class = detector_param["hand_class"]
    logger.info("> hand_class = {}".format(hand_class))

    logger.info("> setup pose")
    pose_model, pose_param = setup_pose(args)
    logger.info("> set up camera, cameraId = {}".format(args.camera))
    cap = cv2.VideoCapture(args.camera)
    if cap.isOpened() is False:
        raise Exception("Error opening video stream of file")
    logger.info("> camera check")
    ret_val, image = cap.read()
    if not ret_val:
        raise Exception("camera error")
    logger.info("> finish setup")
    logger.info("> start demo")
    fps_time = 0
    while cap.isOpened():
        ret_val, image = cap.read()
        # convert color BGR -> RGB and HWC -> CHW
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        _, cH, cW = image.shape
        sz = min(cH, cW)
        image = chainercv.transforms.center_crop(image, (sz, sz))
        _, inH, inW = image.shape
        canvasH, canvasW = inH, 2 * inW // 3
        projboxH = canvasH // 3
        projboxW = canvasW // 2
        logger.info("> detect hand")
        time_detect = time.time()
        # detect hand
        bboxes, labels, scores = detector_model.predict(
            np.expand_dims(
                image,
                axis=0
            )
        )
        bbox, label, score = bboxes[0], labels[0], scores[0]
        logger.info("> detect hand {} [msec]".format(1000 * (time.time() - time_detect)))
        pred_vu = []
        proj_pts = []
        found = {
            "left": False,
            "right": False,
            "hand": False,
        }
        for i, (b, l, s) in enumerate(zip(bbox, label, score)):
            ymin, xmin, ymax, xmax = b.astype(int)
            ulen = xmax - xmin
            vlen = ymax - ymin
            boxscale = 1.
            boxlen = int(boxscale * max(ulen, vlen))
            uc = int((xmax + xmin) / 2)
            vc = int((ymax + ymin) / 2)
            umin = max(0, uc - boxlen // 2)
            umax = min(inW, uc + boxlen // 2)
            vmin = max(0, vc - boxlen // 2)
            vmax = min(inH, vc + boxlen // 2)
            patch = image[:, vmin:vmax, umin:umax]
            _, oriH, oriW = patch.shape
            patch = chainercv.transforms.resize(patch, (pose_model.inH, pose_model.inW))
            name = hand_class[l]
            if name == "left":
                # flip
                patch = chainercv.transforms.flip(patch, x_flip=True)

            kp_vu, kp_zyx = get_result_ppn(patch, pose_model, pose_param)
            if name == "left":
                kp_vu = chainercv.transforms.flip_point(
                    kp_vu[np.newaxis],
                    (pose_model.inH, pose_model.inW),
                    x_flip=True,
                ).squeeze(axis=0)

            kp_proj = [kp_zyx[:, [1, 2]], kp_zyx[:, [1, 0]], kp_zyx[:, [0, 2]]]
            if hand_class == ["left", "right"]:
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
            kp_vu = np.array([[oriH / pose_model.inH, oriW / pose_model.inW]]) * kp_vu + np.array([[vmin, umin]])
            if hand_class == ["left", "right"]:
                if found[name]:
                    continue
                else:
                    pred_vu.append(kp_vu)
                    found[name] = True
            else:
                pred_vu.append(kp_vu)

        time_draw = time.time()
        # CHW -> HWC
        image = image.transpose(1, 2, 0)
        pil_image = Image.fromarray(image.astype(np.uint8))
        pil_canvas = Image.new("RGB", (canvasW, canvasH), (0, 0, 0))
        draw_bbox(pil_image, bbox, label, hand_class, DETECTOR_COLOR_MAP)
        canvas_bbox = np.array([
            [0, 0, projboxH, projboxW],
            [projboxH, 0, 2 * projboxH, projboxW],
            [2 * projboxH, 0, 3 * projboxH, projboxW],
            [0, projboxW, projboxH, 2 * projboxW],
            [projboxH, projboxW, 2 * projboxH, 2 * projboxW],
            [2 * projboxH, projboxW, 3 * projboxH, 2 * projboxW],
        ])
        if hand_class == ["left", "right"]:
            canvas_label = [1, 1, 1, 0, 0, 0]
        else:
            canvas_label = [0, 0, 0, 0, 0, 0]
        draw_bbox(pil_canvas, canvas_bbox, canvas_label, hand_class)
        draw_hands(pil_image, pred_vu, pose_model.edges)
        proj_pts = proj_pts[:6]  # proj hand domain with shape 3x2
        draw_hands(pil_canvas, proj_pts, pose_model.edges)
        logger.info("> draw result {} [msec]".format(1000 * (time.time() - time_draw)))
        cv2.namedWindow("HandDetectorDemo", cv2.WINDOW_AUTOSIZE)
        image = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
        canvas = np.asarray(pil_canvas)
        image_show = np.concatenate([image, canvas], axis=1)
        msg = 'GPU ON' if chainer.backends.cuda.available else 'GPU OFF'
        cv2.putText(image_show, 'FPS: % f' % (1.0 / (time.time() - fps_time)),
                    (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("HandDetectorDemo {}".format(msg), image_show)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:  # press ESC to stop
            break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("detector", type=str)
    parser.add_argument("pose", type=str)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    with chainer.using_config("autotune", True), \
         chainer.using_config("use_ideep", 'auto'), \
         chainer.function.no_backprop_mode():
        main(args)
