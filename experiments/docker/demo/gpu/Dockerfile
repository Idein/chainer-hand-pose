# Docker image for demo
# Reference:
# https://github.com/chainer/chainer/tree/master/docker

FROM nvidia/cuda:10.0-cudnn7-devel
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-wheel \
    python3-setuptools \
    python3-tk \
    git \
    g++ \
    make \
    cmake \
    libblas3 \
    libblas-dev \
    python-opencv \
    curl ca-certificates \
    && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# upgrade pip
RUN curl -kL https://bootstrap.pypa.io/get-pip.py | python3
# install python packages
RUN pip3 install --no-cache-dir numpy scipy matplotlib \
    pillow opencv-python && \
    pip3 install --no-cache-dir \
    chainer==6.2.0 cupy-cuda100==6.2.0 chainercv==0.13.0 ideep4py

# https://github.com/P0cL4bs/WiFi-Pumpkin/issues/53
ENV QT_X11_NO_MITSHM=1
# Use Agg backend for matplotlib
ENV DISPLAY 0
