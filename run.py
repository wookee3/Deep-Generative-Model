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
    
    args = parser.parse_args()    
    if args.dataset == 'cifar10':
        factor = CIFAR10_NORMALIZE_FACTOR
        dataset = torchvision.datasets.CIFAR10
        dest = 'data/cifar10'
    elif args.dataset == 'mnist':
        factor = MNIST_NORMALIZE_FACTOR
        dataset = torchvision.datasets.MNIST
        dest = 'data/mnist'
    elif args.dataset == 'celeba':
        factor = CELEBA_NORMALIZE_FACTOR
    else:
        raise RuntimeError("choose dataset among cifar10 and mnist")

    transform = torchvision.transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(factor[0], factor[1])]
    )
    
