from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder, MNIST, CIFAR10
from PIL import Image
import torch
import numpy as np
import os, sys
import random
import requests
from six.moves import urllib
from utils.constant import NORMALIZE_FACTOR, CROP_SIZE, IMAGE_SIZE
from utils.data.augmentation import RetinaAugmenter


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
        if label is None:
            self.idxs = [i for i in range(len(self.targets))]
        else:
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


class PneumoniaDataset(data.Dataset):
    def __init__(self, transform=None, dir_path=None, is_train=True):
        self.transform = transform
        self.targets = dict()
        self.name = 'RSNA'
        self.dir_path = os.path.join(dir_path, 'training') if is_train else os.path.join(dir_path, 'validation')

        # add to id list
        with open(os.path.join(self.dir_path, 'labels.csv'), 'r') as f:
            for line in f.readlines():
                splitted = line.strip().split(',')
                patient_id = splitted[0]
                label_coord = splitted[1:-1]

                if not label_coord[0]:
                    self.targets[patient_id] = []
                    continue

                if not is_train:
                    label_coord = [float(s) for s in label_coord]
                    # width, height to xmax, ymax
                    label_coord[2] += label_coord[0]
                    label_coord[3] += label_coord[1]
                    label_id = splitted[-1]
                    if patient_id in self.targets.keys():
                        self.targets[patient_id] = self.targets[patient_id] + [(label_coord, label_id)]
                    else:
                        self.targets[patient_id] = [(label_coord, label_id)]
            f.close()
        self.ids = sorted(list(self.targets.keys()))

    def __getitem__(self, index):
        # im: image
        # gt: lists of [xmin, ymin, xmax, ymax, label_ind]
        # h : original image height, w : original image width
        im, gt, h, w = self.pull_item_dict(index)
        return {'img': im, 'label': gt}

    def __len__(self):
        return len(self.ids)

    def pull_item_dict(self, index):
        patient_id = self.ids[index]
        targets = self.targets[patient_id]
        loc_array = []
        label_array = []

        for target in targets:
            loc_array.append(target[0])
            label_array.append(target[1])

        img = np.load(os.path.join(self.dir_path, '%s.npy' % patient_id)).astype(np.float32)
        height, width = img.shape
        locs = np.array(loc_array, dtype=np.float32)
        labels = np.atleast_1d(label_array).astype(np.float32)

        # for object detection
        if len(targets) == 0:
            locs = np.empty(shape=(0,4))
            labels = np.empty(shape=(0,))
        if locs.ndim == 1:
            locs = np.expand_dims(locs, axis=0)

        if self.transform is not None:
            img, locs, labels = self.transform(img, locs, labels)

        gt = np.concatenate((locs, np.expand_dims(labels, axis=1) - 1.), axis=1)

        return img, gt, height, width


def get_loader(train, name, root_path, loader,
               label=None, selected_attrs=None):
    """Build and return a data loader."""
    dataset_name = name.lower()

    # build transform
    transform = []
    factor = NORMALIZE_FACTOR[dataset_name]
    if dataset_name == 'celeba':
        if train:
            transform.append(T.RandomHorizontalFlip())
        transform.append(T.CenterCrop(CROP_SIZE))
        transform.append(T.Resize(IMAGE_SIZE))
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=factor[0], std=factor[1]))
    else:
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=factor[0], std=factor[1]))
    transform = T.Compose(transform)

    # make dataset
    root_dir = os.path.join(root_path, name)
    if dataset_name == 'celeba':
        dataset = CelebA(root_dir, selected_attrs, transform, train)
    elif dataset_name == 'cifar10':
        dataset = CIFAR10(root_dir, train, transform, download=True)
    elif dataset_name == 'mnist':
        if label is None:
            dataset = MNIST(root_dir, train, transform, download=True)
        else:
            dataset = LabelMnist(root_dir, label, train, transform, download=True)
    elif dataset_name == 'imagenet':
        dataset = ImageFolder(root_dir, transform)
    elif dataset_name == 'pneumonia':
        augmenter = RetinaAugmenter(1024, train)
        dataset = PneumoniaDataset(augmenter, root_dir)
    else:
        raise RuntimeError("not provided dataset")

    # make dataloader
    data_loader = data.DataLoader(dataset=dataset, **loader)

    return data_loader


if __name__ == '__main__':
    # transform = T.Compose([T.ToTensor()])
    
    # dataset = LabelMnist('data/mnist', 1, True, transform, download=True)
    # loader = data.DataLoader(dataset=dataset,
    #                          batch_size=16,
    #                          shuffle=True)

    # pneumonia dataset test
    transform = RetinaTransform(1024, False)
    root_path = 'D:\\dataset\\image\\pneumonia\\training'
    dataset = RSNADataset(transform, root_path, is_normal=False)
    print(dataset[0])
    print(dataset[1])
    print(dataset[4])