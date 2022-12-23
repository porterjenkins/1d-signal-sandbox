import numpy as np


class RunningAvgQueue(object):
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.data=[]

    def __str__(self):
        return str(self.data)

    def add(self, x):
        self.data.append(x)
        if len(self.data) > self.maxsize:
            self.data.pop(0)

    def mean(self):
        return np.mean(self.data)


def get_n_params(model):
    n_params = 0
    for layer in model.parameters():
        dims = layer.size()
        cnt = dims[0]
        for d in dims[1:]:
            cnt *= d
        n_params += cnt

    return n_params


