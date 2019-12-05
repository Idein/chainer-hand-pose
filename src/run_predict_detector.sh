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
    # Reference
    # http://wiki.ros.org/docker/Tutorials/GUI
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
