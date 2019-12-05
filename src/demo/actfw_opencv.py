from actfw.task import Task, Producer, Consumer
from actfw.capture import Frame
import cv2
import time
import traceback
import signal
import logging
import queue

logger = logging.getLogger(__name__)


class OpenCVPreprocessor():
    def __init__(self, inW, inH, color="BGR", layout="HWC"):
        self.inW = inW
        self.inH = inH
        if not color in ["RGB", "BGR", "GRAY"]:
            raise ValueError(
                "color must be 'RGB','BGR' or 'GRAY', actual {}".format(color)
            )
        self.color = color
        if not layout in ["HWC", "CHW"]:
            raise ValueError(
                "layout must be 'HWC' or 'CHW' actual {}".format(layout)
            )
        self.layout = layout

    def proc(self, img):
        raise NotImplementedError("please implement proc function")


class CenterCropScale(OpenCVPreprocessor):
    def __init__(self, *args, **kwargs):
        super(CenterCropScale, self).__init__(*args, **kwargs)

    def proc(self, img):
        # assume image layout of `img` should be "HWC"
        if self.color == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.color == "GRAY":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("invalid color {}".format(self.color))
        H, W, _ = img.shape
        if self.inH > H or self.inW > W:
            raise ValueError('shape of image needs to be larger than size')
        aspect = min(H, W) / max(self.inH, self.inW)

        cropH = int(aspect * self.inH)
        cropW = int(aspect * self.inW)

        y_offset = int(round(H - cropH) / 2.)
        x_offset = int(round(W - cropW) / 2.)

        y_slice = slice(y_offset, y_offset + cropH)
        x_slice = slice(x_offset, x_offset + cropW)
        img = img[y_slice, x_slice, :]
        img = cv2.resize(img, (self.inW, self.inH))
        return img


class OpenCVCamera(Producer):
    def __init__(self, preprocessor, capture=0):
        super(OpenCVCamera, self).__init__()
        """use OpenCV camera"""
        self.capture = capture
        self.preprocessor = preprocessor
        self.color = preprocessor.color
        self.frames = []

    def run(self):
        """Run producer activity"""
        while self._is_running():
            try:
                ret_val, img = self.capture.read()
                if not ret_val or img is None:
                    break
                img = self.preprocessor.proc(img)
                updated = 0
                for frame in reversed(self.frames):
                    if frame._update(img):
                        updated += 1
                    else:
                        break
                self.frames = self.frames[len(self.frames) - updated:]

                frame = Frame(img)
                if self._outlet(frame):
                    self.frames.append(frame)

            except Exception as e:
                logger.error("error in {} with error \n{}".format(self, e))
                traceback.print_exc()
                # notify stop event to parent application
                self.app.running = False
                break

    def _outlet(self, o):
        length = len(self.out_queues)
        while self._is_running():
            try:
                self.out_queues[self.out_queue_id % length].put(o, block=False)
                self.out_queue_id = (self.out_queue_id + 1) % length
            except queue.Full:
                pass
            except queue.Empty:
                pass
            return True


class Viewer(Consumer):

    def __init__(self, color):
        super(Viewer, self).__init__()
        self.color = color
        self.img = None

    def proc(self, frame):
        img = frame.getvalue()
        self.img = img


class OpenCVApplication:
    """Actcast Application"""

    def __init__(self, capture_color="RGB"):
        self.running = True
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)
        self.capture_color = capture_color
        self.tasks = []

    def _handler(self, sig, frame):
        self.running = False

    def register_task(self, task):
        """

        Register the application task.

        Args:
            task (:class:`~actfw.task.task.Task`): task
        """
        logger.info("> {}".format(type(task)))
        if not issubclass(type(task), Task):
            raise TypeError(
                "type(task) must be a subclass of actfw.task.Task."
            )
        task.app = self
        self.tasks.append(task)

    def run(self):
        """Start application"""
        for task in self.tasks:
            logger.info("> {}".format(type(task)))
            task.start()
        try:
            fps_time = time.time()
            while self.running:
                img = self.tasks[-1].img
                if img is not None:
                    if self.capture_color == "RGB":
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.putText(img, 'FPS: % f' % (1.0 / (time.time() - fps_time)),
                                (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.imshow("ApplicationName", img)
                    # reset state
                    fps_time = time.time()
                    self.tasks[-1].img = None
                if cv2.waitKey(1) == 27:
                    break
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.error("{} {}".format(self.__class__.__name__, e))

        for task in self.tasks:
            task.stop()
        for task in self.tasks:
            task.join()


def main():
    preprocessor = CenterCropScale(inW=640, inH=480, color="RGB")
    capture = cv2.VideoCapture(0)
    ret_val, img = capture.read()
    if not ret_val:
        raise Exception("SOMETHING WRONG")
    cam = OpenCVCamera(preprocessor, capture)
    viewer = Viewer(cam.color)
    cam.connect(viewer)
    app = OpenCVApplication(capture_color=cam.color)
    app.register_task(cam)
    app.register_task(viewer)
    app.run()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
