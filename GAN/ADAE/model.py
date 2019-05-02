import torch
import torch.nn as nn

from utils.loggings import logger


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

