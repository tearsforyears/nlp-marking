# coding=utf-8
import time

__ver__ = 0.1
__github__ = 'tearsforyears'


def runtime(note=None):
    '''
        a decorator for runtime print
    '''

    def fnn(f):
        def fn(*arg, **kw):
            tic = time.time()
            res = f(*arg, **kw)
            tok = time.time()
            print(note, tok - tic, "s")
            return res

        return fn

    return fnn


class StopWatch(object):
    '''
        is a simple stopwatch for test
    '''

    def __init__(self):
        self.tic = 0
        self.tok = 0
        self.pausetime = 0

    def pause(self, sec):
        time.sleep(sec)
        self.pausetime += sec

    def begin(self):
        self.tic = time.time()

    def stop(self, showtime=True, _continue=True):
        self.tok = time.time()
        if showtime:
            print(self.get_time(), 'ms')
        if _continue:
            self.reset()
            self.begin()

    def reset(self):
        self.tic = 0
        self.tok = 0
        self.pausetime = 0

    def get_time(self):
        return (self.tok - self.tic - self.pausetime) * 1000


def main():
    import numpy as np

    def compute():
        return np.linalg.svd(np.random.rand(1000, 1000))

    sw = StopWatch()
    sw.begin()
    for i in range(10):
        compute()
        sw.stop()


if __name__ == '__main__':
    main()
