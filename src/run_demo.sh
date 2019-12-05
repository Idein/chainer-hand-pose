# Demo script
# You need build docker image from Dockerfile located at
# /path/to/repository/experiments/docker/gpu

docker build -t hand_demo ../experiments/docker/demo/gpu/

CMDNAME=`basename $0`
BASEMODELDIR=$(pwd)/../result

xhost +local:docker
docker run --rm \
--gpus all \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix/:/tmp/.X11-unix \
-v $(pwd):/work \
-v $BASEMODELDIR:/models \
--device=/dev/video0:/dev/video0 \
-w /work \
hand_demo:latest python3 demo.py \
/models/release \
/models/release
xhost -local:docker
