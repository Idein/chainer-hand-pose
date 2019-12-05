from actfw.task import Pipe


class Sequential(Pipe):
    def __init__(self, tasks):
        super(Sequential, self).__init__()
        self.tasks = tasks

    def register_app(self, app):
        for t in self.tasks:
            t.app = app

    def proc(self, frame):
        pipe = frame
        for t in self.tasks:
            pipe = t.proc(pipe)
        return pipe
