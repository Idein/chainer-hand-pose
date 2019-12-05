# coding: utf-8
# Object Detection Demo
import argparse
import cv2
import numpy as np
import os
import sys
import time
import tensorflow as tf

from distutils.version import StrictVersion

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

# Path to label and frozen detection graph. This is the actual model that is used for the object detection.
parser = argparse.ArgumentParser(description='object_detection_tutorial.')
parser.add_argument("trained", type=str)
parser.add_argument('-m', '--model', default='frozen_inference_graph.pb')
parser.add_argument('-d', '--device', default='normal_cam')  # normal_cam / jetson_nano_raspi_cam / jetson_nano_web_cam
parser.add_argument("--camera", type=int, default=0)
args = parser.parse_args()

detection_graph = tf.Graph()

mode = 'bbox'


def load_graph():
    """
    download frozen_inference_graph.pb
    $ wget https://raw.githubusercontent.com/victordibia/handtracking/master/model-checkpoint/ssdlitemobilenetv2/frozen_inference_graph.pb
    """
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(args.model, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        return detection_graph


# Load a (frozen) Tensorflow model into memory.
print('Loading graph...')
detection_graph = load_graph()
print('Graph is loaded')

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
with detection_graph.as_default():
    tf_sess = tf.Session(config=tf_config)
    ops = tf.compat.v1.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                tensor_name)

    image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')


def run_inference_for_single_image(image, graph):
    # Run inference
    output_dict = tf_sess.run(
        tensor_dict,
        feed_dict={image_tensor: image},
    )

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.int64)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    return output_dict


# Switch camera according to device
if args.device == 'normal_cam':
    cam = cv2.VideoCapture(args.camera)
elif args.device == 'jetson_nano_raspi_cam':
    GST_STR = 'nvarguscamerasrc \
    ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=(fraction)30/1 \
    ! nvvidconv ! video/x-raw, width=(int)1920, height=(int)1080, format=(string)BGRx \
    ! videoconvert \
    ! appsink'
    cam = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER)  # Raspi cam
elif args.device == 'jetson_nano_web_cam':
    cam = cv2.VideoCapture(1)
else:
    print('wrong device')
    sys.exit()

import argparse
import configparser
import logging

logger = logging.getLogger()
import os

import cv2

import chainer
import chainercv

import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from models.selector import select_model
from hand_dataset.selector import select_dataset
from hand_dataset.image_utils import normalize_rgb

count_max = 0

logging.basicConfig(level=logging.INFO)

config = configparser.ConfigParser()

path = os.path.expanduser(os.path.join(args.trained, "result", "config.ini"))
logger.info("read {}".format(path))
config.read(path, 'UTF-8')

logger.info("setup devices")
chainer.global_config.autotune = True
chainer.config.cudnn_fast_batch_normalization = True

# dataset_type = config["dataset"]["type"]
use_rgb = config.getboolean("dataset", "use_rgb")
use_depth = config.getboolean("dataset", "use_depth")
assert use_rgb
assert use_rgb ^ use_depth, "XOR(use_rgb, use_depth) must be True"
hand_param = select_dataset(config, return_data=["hand_param"])
model_path = os.path.expanduser(os.path.join(args.trained, "result", "bestmodel.npz"))

logger.info("> restore model")
model = select_model(config, hand_param)
logger.info("> model.device = {}".format(model.device))

logger.info("> restore models")
chainer.serializers.load_npz(model_path, model)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121)
ax3 = fig.add_subplot(122, projection="3d")
color_map = hand_param["color_map"]
color = [color_map[k] for k in hand_param["keypoint_names"]]
edge_color = [color_map[s, t] for s, t in hand_param["edges"]]
pred_color = [[255, 255, 255] for k in hand_param["keypoint_names"]]

NUM_KEYPOINTS = 21

KEYPOINT_NAMES = []

for k in ["wrist", "thumb", "index", "middle", "ring", "little"]:
    if k == "wrist":
        joint_name = "_".join([k])
        KEYPOINT_NAMES.append(joint_name)
    else:
        for p in ["tip", "dip", "pip", "mcp"]:
            joint_name = "_".join([k, p])
            KEYPOINT_NAMES.append(joint_name)

EDGE_NAMES = []
from utils import pairwise

for f in ["index", "middle", "ring", "little", "thumb"]:
    for p, q in pairwise(["wrist", "mcp", "pip", "dip", "tip"]):
        if p != "wrist":
            p = "_".join([f, p])
        q = "_".join([f, q])
        EDGE_NAMES.append([p, q])

EDGES = [[KEYPOINT_NAMES.index(s), KEYPOINT_NAMES.index(t)]
         for s, t in EDGE_NAMES]

from hand_dataset.common_dataset import EDGES


def ch_inferenec(image):
    image = cv2.resize(image, (hand_param["inW"] // 4, hand_param["inH"] // 4))
    image = image.transpose(2, 0, 1)  # HWC -> CHW
    image = chainercv.transforms.resize(image, (hand_param["inH"], hand_param["inW"]))
    ret = model.predict(np.expand_dims(normalize_rgb(image), axis=0))
    if len(ret) == 7:
        resp, conf, x, y, w, h, v = ret
    else:
        resp, conf, x, y, w, h, e, v = ret
    resp = chainer.backends.cuda.to_cpu(resp.array)
    conf = chainer.backends.cuda.to_cpu(conf.array)
    w = chainer.backends.cuda.to_cpu(w.array)
    h = chainer.backends.cuda.to_cpu(h.array)
    x = chainer.backends.cuda.to_cpu(x.array)
    y = chainer.backends.cuda.to_cpu(y.array)
    # e = chainer.backends.cuda.to_cpu(e.array)
    v = chainer.backends.cuda.to_cpu(v.array)
    resp = np.squeeze(resp, axis=0)
    conf = np.squeeze(conf, axis=0)
    x = np.squeeze(x, axis=0)
    y = np.squeeze(y, axis=0)
    w = np.squeeze(w, axis=0)
    h = np.squeeze(h, axis=0)
    # e = np.squeeze(e, axis=0)
    v = np.squeeze(v, axis=0)
    color_map = hand_param["color_map"]
    keypoint_names = hand_param["keypoint_names"]
    edges = hand_param["edges"]
    delta = resp * conf
    scaleH = hand_param["inH"] / model.outsize[1]
    scaleW = hand_param["inW"] / model.outsize[0]
    joint2d = {}
    grid_position = {}
    finger_order = ["mcp", "pip", "dip", "tip"]
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
            sz = 1 if q == ["tip", "dip"] else 2
            hslice = slice(max(0, p_h - sz), min(model.outsize[1], p_h + sz + 1))
            wslice = slice(max(0, p_w - sz), min(model.outsize[0], p_w + sz + 1))
            target = delta[i][hslice, wslice]
            q_h, q_w = np.unravel_index(np.argmax(target), target.shape)
            y_offset = (p_h - sz) + q_h if p_h - sz >= 0 else q_h
            x_offset = (p_w - sz) + q_w if p_w - sz >= 0 else q_w
            joint2d[f_q] = [
                scaleH * (y_offset + y[i][(y_offset, x_offset)]),
                scaleW * (x_offset + x[i][(y_offset, x_offset)])
            ]
            grid_position[f_q] = (y_offset, x_offset)

    kp_zyx = np.zeros((len(keypoint_names), 3))

    for ei, (s, t) in enumerate(edges):
        u_ind = grid_position[keypoint_names[s]]
        orien = v[ei, :, u_ind[0], u_ind[1]]
        elen = 1.5 if s == 0 else 1
        kp_zyx[t] = kp_zyx[s] + orien * elen

    joint2d = np.array([joint2d[k] for k in keypoint_names])
    return joint2d


if __name__ == '__main__':
    count = 0

    labels = ['blank', 'hand']

    while True:
        ret, img = cam.read()
        if not ret:
            print('error')
            break
        key = cv2.waitKey(1)

        if key == 27:  # when ESC key is pressed break
            break

        count += 1
        if count > count_max:
            img_bgr = cv2.resize(img, (300, 300))

            # convert bgr to rgb
            image_np = img_bgr[:, :, ::-1]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            start = time.time()
            output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
            elapsed_time = time.time() - start

            for i in range(output_dict['num_detections']):
                class_id = output_dict['detection_classes'][i]
                if class_id < len(labels):
                    label = labels[class_id]
                else:
                    label = 'unknown'

                detection_score = output_dict['detection_scores'][i]

                if detection_score > 0.5:
                    # Define bounding box
                    h, w, c = img.shape
                    box = output_dict['detection_boxes'][i] * np.array(
                        [h, w, h, w])
                    ymin, xmin, ymax, xmax = box.astype(int)
                    ulen = xmax - xmin
                    vlen = ymax - ymin
                    boxscale = 1.5
                    boxlen = int(boxscale * max(ulen, vlen))
                    uc = int((xmax + xmin) / 2)
                    vc = int((ymax + ymin) / 2)
                    umin = max(0, uc - boxlen // 2)
                    umax = min(w, uc + boxlen // 2)
                    vmin = max(0, vc - boxlen // 2)
                    vmax = min(h, vc + boxlen // 2)
                    crop_img = img[vmin:vmax, umin:umax][:, :, ::-1]
                    oriH, oriW, _ = crop_img.shape
                    joint2d = ch_inferenec(crop_img)
                    if joint2d is not None:
                        joint2d = np.array([[oriH / model.inH, oriW / model.inW]]) * joint2d + np.array([[vmin, umin]])

                        for v, u in joint2d:
                            cv2.circle(img, (int(u), int(v)), 3, (255, 0, 0), -1)
                        for s, t in EDGES:
                            sy, sx = joint2d[s].astype(int)
                            ty, tx = joint2d[t].astype(int)
                            cv2.line(img, (sx, sy), (tx, ty), (255, 0, 0), 5)
                        speed_info = '%s: %f' % ('speed=', elapsed_time)
                        cv2.putText(img, speed_info, (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.rectangle(img,
                                      (umin, vmin), (umax, vmax), (0, 255, 0), 3)
                    else:
                        # Draw bounding box
                        cv2.rectangle(img,
                                      (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)

                    # Put label near bounding box
                    information = '%s: %f' % (label, output_dict['detection_scores'][i])
                    cv2.putText(img, information, (xmin, ymax),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow('detection result', img)
            count = 0

    tf_sess.close()
    cam.release()
    cv2.destroyAllWindows()
