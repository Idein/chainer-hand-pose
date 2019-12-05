# -*- coding: utf-8 -*-

import glob
from importlib import import_module
import os
import random

import chainer
import numpy as np
import shutil

from detector.models.ssd_v2 import SSD


def parse_kwargs(args):
    if args == '':
        return {}

    kwargs = {}
    for arg in args.split(','):
        key, value = arg.split('=')
        kwargs[key] = value

    return kwargs


def setup_devices(ids):
    """setup GPU devices for training"""

    if ids == '':
        # CPU mode
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


def create_result_dir(result_dir):
    target_list = [
        ["", ["*.py", "*.sh", "*.ini"]],
        ["detector",["*.py", "*.sh", "*.ini"]],
        [os.path.join("detector","hand_dataset"), ["*.py"]],
        [os.path.join("detector","graphics"), ["*.py"]],
        [os.path.join("detector","models"), ["*.py"]],
        [os.path.join("detector","evaluation"), ["*.py"]],
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


def get_config(config):
    param = {}

    # training param
    param['batchsize'] = config.getint('training_param', 'batchsize')
    param['learning_rate'] = config.getfloat('training_param', 'learning_rate')
    param['gpus'] = config.get('training_param', 'gpus')
    param['num_process'] = config.getint('training_param', 'num_process')
    param['seed'] = config.getint('training_param', 'seed')
    param['train_iter'] = config.getint('training_param', 'train_iter')
    schedule = config.get('training_param', 'schedule').split(",")
    schedule = list(map(lambda x: int(x.strip()), schedule))
    param['schedule'] = schedule

    # model param
    param['model_path'] = config.get('model_param', 'model_path')
    param['feature_extractor'] = config.get('model_param', 'feature_extractor')
    param['ssd_extractor'] = config.get('model_param', 'ssd_extractor')
    param['input_size'] = config.getint('model_param', 'input_size')
    param['num_layers'] = config.getint('model_param', 'num_layers')
    hand_class = config.get('model_param', 'hand_class').split(",")
    hand_class = tuple(k.strip() for k in hand_class)
    param['hand_class'] = hand_class
    param['smin'] = config.getfloat('model_param', 'smin')
    param['smax'] = config.getfloat('model_param', 'smax')
    param['width_multiplier'] = config.getfloat('model_param', 'width_multiplier')
    param['resolution_multiplier'] = config.getfloat('model_param', 'resolution_multiplier')

    return param


def create_ssd_model(train_param):
    model_path = '.'.join(os.path.split(train_param['model_path']))
    model = import_module("detector."+model_path)
    feature_extractor = getattr(model, train_param['feature_extractor'])(train_param['width_multiplier'])
    ssd_extractor = getattr(model, train_param['ssd_extractor'])(feature_extractor, train_param['width_multiplier'])
    model = SSD(
        ssd_extractor, input_size=int(train_param['input_size'] * train_param['resolution_multiplier']),
        n_fg_class=len(train_param['hand_class']),
        num_layers=train_param['num_layers'],
        smin=train_param['smin'], smax=train_param['smax'],
        # aspect_ratios=((2,), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3)),
        aspect_ratios=((1,), (1,), (1,), (1,), (1,), (1,)),
        variance=(0.1, 0.2), mean=0
    )
    return model
