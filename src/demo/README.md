# Demo script using `actfw`

- This is an exmaple of hand pose demo that utilizes [actfw](https://pypi.org/project/actfw/).
- `actfw` is a Python API for developing [Actcast](https://actcast.io/) apps with a task parallel model.
- We provide [actfw_opencv.py](actfw_opencv.py) that implements subclass of [Producer](https://github.com/Idein/actcast-app-python/blob/master/actfw/task/producer.py) and [Consumer](https://github.com/Idein/actcast-app-python/blob/master/actfw/task/consumer.py) that uses camera API of OpenCV. It will help developer who would like to develop/debug Actcast application on your PC rather than Raspberry Pi.


# Basic usage

```
$ pip install actfw
$ pip install -e ../../ # install our `hand` package
$ python3 demo.py
```

# Convert Chainer to NNOIR format

- We have a computational graph compiler which accelerates deep learning inferences using GPU of Raspberry Pi.
- The Raspberry Pi series uses a GPU called VideoCore IV (VC4) to render on display. Displaying is not necessary in most cases if we use Raspberry Pis as sensing devices. Therefore we use vacant VC4C to accelerate deep-learning inferences.
- Our compiler generates C code that utilizes VC4 as an accelerator for deep learning inferences with the following steps:
  - Step 1. Extract computational graph from pre-trained model of Chainer or ONNX format. See our repository [Idein/nnoir](https://github.com/Idein/nnoir).
  - Step 2. Convert the computational graph to `NNOIR`(= NN Optimization IR) format.
  - Step 3. Generate C code for V4C and compile to shared library.
- Here, we provide scripts [convert_detector.py](convert_detector.py) and [convert_pose.py](convert_pose.py) that convert pre-trained Chainer model to NNOIR format (Step 1 and Step 2).

```
$ python3 convert_pose.py ../../result/release
$ python3 convert_detector.py ../../result/release
```

- If you like to go to the next step, you are supposed to participate a `Actcast Partners`. It is a partner program that aims to support companies or
organizations who want to develop valuable solutions and do business with
Actcast. See the following link to learn more:
  - https://actcast.io/docs/files/partner_program.pdf
