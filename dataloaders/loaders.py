import torch
import torch.utils.data as data
import numpy as np
import os, sys
import random


class RSNADetection(data.Dataset):
    def __init__(self, transform=None, dir_path=None):
        self.transform = transform
        self.targets = dict()
        self.name = 'RSNA'
        self.dir_path = dir_path
        # add to id list
        with open(os.path.join(dir_path, 'labels.csv'), 'r') as f:
            for line in f.readlines():
                splitted = line.strip().split(',')
                patient_id = splitted[0]
                label_coord = splitted[1:-1]

                if not label_coord[0]:
                    self.targets[patient_id] = []
                    continue

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

        if len(targets) == 0:
            locs = np.empty(shape=(0,4))
            labels = np.empty(shape=(0,))
        if locs.ndim == 1:
            locs = np.expand_dims(locs, axis=0)

        if self.transform is not None:
            img, locs, labels = self.transform(img, locs, labels)

        gt = np.concatenate((locs, np.expand_dims(labels, axis=1) - 1.), axis=1)

        return img, gt, height, width


def collate_fn(batch):
    img = []
    label = []

    for sample in batch:
        img.append(sample['img'])
        label.append(sample['label'])

    return img, label