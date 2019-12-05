cat config_pose.ini
docker run --rm \
	-v $(pwd):/work \
	-v ~/dataset:/root/dataset \
	--name hand_pose \
	-w /work idein/chainer:6.2.0 python3 train_pose.py --config-path config_pose.ini
