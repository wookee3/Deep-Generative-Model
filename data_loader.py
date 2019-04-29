from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder, MNIST, CIFAR10
from util.constant import NORMALIZE_FACTOR
from PIL import Image
import torch
import os, sys
import random
import requests
from six.moves import urllib


def download(url, dirpath):
    filename = url.split('/')[-1]
    filepath = os.path.join(dirpath, filename)
    u = urllib.request.urlopen(url)
    f = open(filepath, 'wb')
    filesize = int(u.headers["Content-Length"])
    print("Downloading: %s Bytes: %s" % (filename, filesize))

    downloaded = 0
    block_sz = 8192
    status_width = 70
    while True:
        buf = u.read(block_sz)
        if not buf:
            print('')
            break
        else:
            print('')
        downloaded += len(buf)
        f.write(buf)
        status = (("[%-" + str(status_width + 1) + "s] %3.2f%%") %
                  ('=' * int(float(downloaded) / filesize * status_width) + '>', downloaded * 100. / filesize))
        print(status)
        sys.stdout.flush()
    f.close()
    return filepath


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = {'id': id}, stream = True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


# StarGan code
class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, root, selected_attrs, transform, train=True, download=False):
        """Initialize and preprocess the CelebA dataset."""
        self.root_path = root
        self.image_dir = os.path.join(root, "images")
        self.attr_path = os.path.join(root, 'Anno', 'list_attr_celeba.txt')
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

    def download(self):
        import zipfile

        # download if not exists
        if os.path.exists(self.root_path, "images"):
            return
        # make path
        if not os.path.exists(self.root_path):
            os.mkdir(self.root_path)

        # download file
        drive_id = "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
        filepath = os.path.join(self.root_path, "img_align_celeba.zip")
        download_file_from_google_drive(drive_id, filepath)

        # unzip
        zip_dir = ''
        with zipfile.ZipFile(filepath) as zf:
            zip_dir = zf.namelist()[0]
            zf.extractall(self.root_path)
        os.remove(filepath)
        os.rename(os.path.join(self.root_path, zip_dir), os.path.join(self.root_path, "images"))

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


class LabelMnist(MNIST):
    def __init__(self, root, label, train=True, transform=None, target_transform=None, download=False):
        super(LabelMnist, self).__init__(root, train, transform, target_transform, download)
        self.idxs = [i for i in range(len(self.targets)) if self.targets[i] == label]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        idx = self.idxs[index]
        img, target = self.data[idx], int(self.targets[idx])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_loader(dataset_name, label=None, selected_attrs=None, crop_size=178, image_size=128,
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
        root_dir = 'data/mnist'
        if label is None:
            dataset = MNIST(root_dir, train, transform, download=True)
        else:
            dataset = LabelMnist(root_dir, label, train, transform, download=True)
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


if __name__ == '__main__':
    transform = T.Compose([T.ToTensor()])
    
    dataset = LabelMnist('data/mnist', 1, True, transform, download=True)
    loader = data.DataLoader(dataset=dataset,
                             batch_size=16,
                             shuffle=True)
