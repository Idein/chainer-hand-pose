# How to train hand detector

- Our hand detector is based on SSD i.e. Single Shot MultiBox Detector.
- Many training scripts related to SSD are taken from [ChainerCV project](https://github.com/chainer/chainercv) which provides library for deep learning in computer vision.
- We can detect/crop human hand including hand side information.

## Edit `src/config_detector.ini`

After downloading dataset e.g. under the directory `~/dataset`, you will configure
`src/config_detector.ini` to specify training parameter, dataset directory or something like that.
Here is an example of our `src/config_detector.ini`.

```
$ cd /path/to/src
$ cat config_detector.ini
[training_param]
# batchsize / num of gpus equals the batchsize per gpu
batchsize = 64
learning_rate = 0.001
gpus = main=0, slave1=1
num_process = 16
seed = 0
train_iter = 200000
schedule = 150000


# input file or dir path in docker container environment
# select dataset
# ConfigParser.getboolean is case-insensitive and recognizes
# Boolean values from
# from `yes`/`no`, `on`/`off`, `true`/`false` and `1`/`0`
[dataset]
enable_x_flip = yes
enable_y_flip = no
angle_range = -30,30
train_set = rhd,stb
val_set = rhd,stb
test_set = stb

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

[output_path]
result_dir = /work/trained

[model_param]
model_path = models/hand_ssd_mobilenet_v2
feature_extractor = MobileNetV2
ssd_extractor = MobileNetV2LiteExtractor300
input_size = 256
num_layers = 6
# must be
#`hand_class=left, right`
# or
#`hand_class=hand`
hand_class = left, right
smin = 0.025
smax = 0.8
width_multiplier = 1.0
resolution_multiplier = 1.0
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
- Please confirm that you set dataset directory correctly.

## run training script

```
$ cd path/to/src
$ cat begin_detector.sh
cat config_detector.ini
docker run --rm \
	--name hand_detector \
	-v $(pwd):/work \
	-v ~/dataset:/root/dataset \
	-w /work idein/chainer:6.2.0 \
	python3 /work/train_detector.py --config_path config_detector.ini
$ sudo bash begin_detector.sh
```


- Note that you'll see a directory named `trained` which (will) contain trained model and source code used on training as a snapshot
- Note that setting option `--gpus all` will be required for some environment to recognize GPU device from docker container.

## run prediction script

prepare `experiments/test_images` folder and store images for prediction.

```
$ cd src
$ cat run_predict_detector.sh
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
    -v $(pwd)/../experiments/test_images:/test_images \
    --name hand_detector \
    -w /work idein/chainer:6.2.0 python3 predict_detector.py \
    --model_path /models/$1/detector/bestmodel.npz \
    --config_path /models/$1/detector/config.ini \
    --img_path /test_images \
    --out_path /models/$1/output
if [[ "$OSTYPE" == "darwin"* ]]; then
    xhost - 127.0.0.1
else [[ "$OSTYPE" == "linux-gnu" ]]
    xhost - local:docker
fi
$ bash run predict_detector.sh trained
```
