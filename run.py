import torch
import torchvision
import torchvision.transforms as transforms
from util.constant import *
from data_loader import get_loader
import argparse


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset among cifar10 and mnist')
    parser.add_argument('--attrs', type=list, default=[], nargs='*')
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()    

    train_loader = get_loader(args.dataset, args.attrs, train=True)

    for batch in train_loader:
        print(batch[0].size())
        print(batch[1].size())
        break