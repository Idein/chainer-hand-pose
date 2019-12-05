import glob
from itertools import tee
import os
import random
import shutil

import chainer
import numpy as np


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def remove_whitespace(string):
    return string.replace(" ", '')


def parse_kwargs(args):
    if args == '':
        return {}

    kwargs = {}
    for arg in args.split(','):
        arg = remove_whitespace(arg)
        key, value = arg.split('=')
        kwargs[key] = value

    return kwargs


def parse_cube(cube, style="DHW"):
    assert sorted(style) == ['D', 'H', 'W'], "variable 'style' must contain D, H and W"

    cube = remove_whitespace(cube)
    cubeW, cubeH, cubeD = list(map(int, cube.split('x')))
    order = {
        'W': cubeW,
        'H': cubeH,
        'D': cubeD,
    }
    cube = np.array([order[w] for w in style])
    return cube


def parse_imsize(imsize, style="HW"):
    imsize = remove_whitespace(imsize)
    imW, imH = list(map(int, imsize.split('x')))
    order = {
        'W': imW,
        'H': imH,
    }
    imsize = np.array([order[w] for w in style])
    return imsize


def setup_devices(ids):
    if ids == '':
        return {'main': -1}
    devices = parse_kwargs(ids)
    for key in devices:
        devices[key] = int(devices[key])
    return devices


def set_random_seed(devices, seed):
    random.seed(seed)
    np.random.seed(seed)
    for key, id in devices.items():
        if id < 0:
            break
        if key == 'main':
            chainer.cuda.get_device_from_id(id).use()
            chainer.cuda.cupy.random.seed(seed)


def save_files(result_dir):
    target_list = [
        ["", ["*.py", "*.sh", "*.ini"]],
        ["pose",["*.py", "*.sh", "*.ini"]],
        [os.path.join("pose","hand_dataset"), ["*.py"]],
        [os.path.join("pose","graphics"), ["*.py"]],
        [os.path.join("pose","models"), ["*.py"]],
        [os.path.join("pose","visualizations"), ["*.py"]],
    ]
    for (folder, patterns) in target_list:
        result_src_dir = os.path.join(result_dir, "src", folder)
        if not os.path.exists(result_src_dir):
            os.makedirs(result_src_dir)
        file_list = []
        for ptn in patterns:
            file_list += glob.glob(os.path.join(folder,ptn))
        for file in file_list:
            shutil.copy(file, os.path.join(result_src_dir, os.path.basename(file)))
