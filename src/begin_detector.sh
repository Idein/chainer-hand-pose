#!/usr/bin/env bash

cat config_detector.ini
docker run --rm \
	--name hand_detector \
	-v $(pwd):/work \
	-v ~/dataset:/root/dataset \
	-w /work idein/chainer:6.2.0 \
	python3 /work/train_detector.py --config_path config_detector.ini
