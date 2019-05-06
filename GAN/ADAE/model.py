import numpy as np
import torch
import torch.nn as nn

from utils.loggings import logger
from .layers import *


def init_weight(m):
    if type(m) == nn.Linear:
        pass
    elif type(m) == nn.Conv3d:
        pass
    elif type(m) == nn.Conv2d:
        pass
    elif type(m) == nn.Conv1d:
        pass
    elif type(m) == nn.ConvTranspose3d:
        pass
    elif type(m) == nn.ConvTranspose2d:
        pass
    elif type(m) == nn.ConvTranspose1d:
        pass
        

class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    def forward(self, batch_data):
        raise NotImplementedError()

        
class ImageEncoder(nn.Module):
    def __init__(self):
        pass

    def forward(self, batch_data):
        pass


class LatentEncoder(nn.Module):
    def __init__(self):
        pass

    def forward(self, batch_data):
        pass


class ImageDecoder(nn.Module):
    def __init__(self):
        pass

    def forward(self, batch_data):
        pass


class LatentDiscriminator(nn.Module):
    def __init__(self):
        pass

    def forward(self, batch_data):
        pass


class ImageDiscriminator(nn.Module):
    def __init__(self):
        pass

    def forward(self, batch_data):
        pass


if __name__ == '__main__':
    logger.info("test for mnist")

