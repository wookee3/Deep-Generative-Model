import torch
import torchvision
import torchvision.transforms as transforms
from util.constant import *
from data_loader import get_loader
from GAN.ADAE.model import Encoder
import argparse


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='choose dataset among cifar10, mnist, celeba')
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()    

    train_loader = get_loader(args.dataset, 0, train=True)

    for batch in train_loader:
        print(batch[0].size())
        print(batch[1])
        break
