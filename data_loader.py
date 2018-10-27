from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder, MNIST, CIFAR10
from util.constant import NORMALIZE_FACTOR
from PIL import Image
import torch
import os
import random


# StarGan code
class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, root, selected_attrs, transform, train=True, download=False):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = os.path.join(root, "images")
        self.attr_path = os.path.join(root, 'list_attr_celeba.txt')
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.train = train
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}

        # download celeba
        if download:
            pass
        
        # preprocess
        self.preprocess()
        
        # divide train and test
        if train:
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.train else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        label = torch.Tensor(label) if label else torch.Tensor([0])

        return self.transform(image), label

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(dataset_name, selected_attrs=None, crop_size=178, image_size=128, 
            batch_size=16, train=True, num_workers=1):
    """Build and return a data loader."""
    dataset_name = dataset_name.lower()

    # build transform
    transform = []
    factor = NORMALIZE_FACTOR[dataset_name]
    if dataset_name == 'celeba':
        if train:
            transform.append(T.RandomHorizontalFlip())
        transform.append(T.CenterCrop(crop_size))
        transform.append(T.Resize(image_size))
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=factor[0], std=factor[1]))
    else:
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=factor[0], std=factor[1]))
    transform = T.Compose(transform)

    # make dataset
    if dataset_name == 'celeba':
        root_dir = 'data/CelebA_nocrop'
        dataset = CelebA(root_dir, selected_attrs, transform, train)
    elif dataset_name == 'cifar10':
        root_dir = 'data/cifar10'
        dataset = CIFAR10(root_dir, train, transform, download=True)
    elif dataset_name == 'mnist':
        dataset = MNIST(root_dir, train, transform, download=True)
    elif dataset_name == 'imagenet':
        root_dir = 'data/imagenet'
        dataset = ImageFolder(root_dir, transform)
    else:
        raise RuntimeError("not provided dataset")

    # make dataloader
    data_loader = data.DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=train,
                                num_workers=num_workers)
    
    return data_loader