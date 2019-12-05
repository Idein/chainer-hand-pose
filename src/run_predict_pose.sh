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
