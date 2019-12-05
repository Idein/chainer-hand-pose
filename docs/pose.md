# How to train hand detector

- Our hand detector is based on SSD i.e. Single Shot MultiBox Detector.
- Many training scripts related to SSD are taken from [ChainerCV project](https://github.com/chainer/chainercv) which provides library for deep learning in computer vision.
- We can detect/crop human hand including hand side class.

## Edit `src/config_pose.ini`

After downloading dataset e.g. under the directory `~/dataset`, you will configure
`src/config_pose.ini` to specify training parameter, dataset directory or something like that.
Here is an example of our `src/config_pose.ini`.

```
$ cat src/config_pose.ini
[training_param]
batch_size = 32
seed = 0
train_iter = 600000
learning_rate = 0.0025
gpus = main = 0, slave = 1
n_processes = 16
init_learning_rate = 0.01
# `2`, `3` or `2,3`. The last option train both 2 and 3 dim
joint_dims = 2,3
# input image size WxH
imsize = 224x224
# input cube DxHxW
cube = 200x200x200
# augmentation parameter
## flip
enable_x_flip = false
enable_y_flip = false
## rotate
angle_range = -90,90
## crop: np.arange(begin,end,step)
scale_range = 1.0, 1.5, 0.01
shift_range = -0.15, 0.15, 0.01


[result]
dir = /work/trained

# select dataset
# ConfigParser.getboolean is case-insensitive and recognizes
# Boolean values from
# from `yes`/`no`, `on`/`off`, `true`/`false` and `1`/`0`
[dataset]
train_set = fhad,stb,multiview,freihand
val_set = fhad,stb,multiview,freihand
test_set = stb
use_rgb = yes
use_depth = no

[dataset_dir]
# First-Person Hand Action Benchmark with RGB-D Videos and 3D Hand Pose Annotations
fhad = ~/dataset/fhad
# Stereo Tracking Benchmark Dataset (STB)
stb = ~/dataset/stb
# RHD
rhd = ~/dataset/RHD_published_v2
# GANerated dataset
ganerated = ~/dataset/GANeratedHands_Release
# Synth Hands dataset
synth = ~/dataset/nyu_hand_dataset_v2
# multiview
multiview = ~/dataset/multiview_hand
# FreiHAND
freihand = ~/dataset/FreiHAND_pub_v1

# select model
[model]
name = ppn

[ppn]
feature_extractor = mv2
# MobileNetV2
[mv2]
width_multiplier = 1.0

# ResNet
[resnet]
# must be `18`, `34` or `50`
n_layers = 18
```

### Tips

- If you have a machine which has multiple GPU and specify GPU id, edit `gpus` option. The device of the name 'main' is used as a "master", while others are used as slaves. Names other than 'main' are arbitrary.
```
gpus = main=0, slave1=1
```
- Of course you can train with a single GPU machine. You are supposed to set the value of `gpus` as follow:
```
gpus = main=0
```

- If you like to train with several dataset on training to increase generalized performance, set the value of `train_set` with sequence of name of dataset. For example
```
[dataset]
train_set = rhd,stb,multiview,freihand
```
- Please confirm that you specify dataset directory correctly.

## Run training script

- Run the following script

```
$ cd src
$ cat begin_train.sh
cat config_pose.ini
docker run --rm \
  --name hand_detector \
	-v $(pwd):/work \
	-v ~/dataset:/root/dataset \
	-w /work idein/chainer:6.2.0 \
	python3 /work/train.py --config_path config_pose.ini
$ sudo bash_begin.sh
```

- Note that you'll see a directory named `trained` which (will) contain trained model and source code used on training as a snapshot
- Note that set option `--gpus all` will be required for some environment to recognize GPU device from docker container.

## Run inference script

After training, Open XQuartz.app and run the following script:

```
$ cd src
$ cat run_predict_pose.sh
#!/usr/bin/env bash

CMDNAME=`basename $0`
BASEMODELDIR=$(pwd)

if [ $# -ne 1 ]; then
    echo "Usage: $CMDNAME path/to/model" 1>&2
    exit 1
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Reference
    # https://medium.com/@mreichelt/how-to-show-x11-windows-within-docker-on-mac-50759f4b65cb
    xhost + 127.0.0.1
    HOSTDISPLAY=host.docker.internal:0
else [[ "$OSTYPE" == "linux-gnu" ]]
    xhost + local:docker
    HOSTDISPLAY=$DISPLAY
fi
docker run --rm \
    -e DISPLAY=$HOSTDISPLAY \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
        -v $BASEMODELDIR:/models \
        -v $(pwd):/work \
        -v ~/dataset:/root/dataset \
        --name hand_pose \
        -w /work idein/chainer:6.2.0 python3 predict_pose.py --trained /models/$1
if [[ "$OSTYPE" == "darwin"* ]]; then
    xhost - 127.0.0.1
else [[ "$OSTYPE" == "linux-gnu" ]]
    xhost - local:docker
fi
$ bash run_predict_pose.sh trained
```

## Evaluation

```
$ python eval_ppn.py trained
```
