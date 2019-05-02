import torch
import torch.nn as nn


def init_weight(m):
    if type(m) == nn.Linear:
        pass
    elif type(m) == nn.Conv2d:
        pass
    elif type(m) == nn.Conv1d:
        pass
    else:
        pass


class Encoder:
    def __init__(self):
        self.name = "enc"

