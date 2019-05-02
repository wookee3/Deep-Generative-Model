import torch
import torch.nn as nn

from utils.loggings import logger


class Encoder:
    def __init__(self):
        self.name = "enc"