from torchvision import transforms
import cv2
import random
import numpy as np
import sys
import transform as ts
import image



class RetinaTransform:
    def __init__(self, size):
        self.min_rotation = -0.05
        self.max_rotation = 0.05
        self.min_translation = (-0.1, -0.1)
        self.max_translation = (0.1, 0.1)
        self.min_shear = 0
        self.max_shear = 0
        self.min_scaling = (0.8, 0.8)
        self.max_scaling = (1.2, 1.2)
        self.flip_x_chance = 0.5
        self.flip_y_chance = 0
        self.prng = ts.DEFAULT_PRNG
        self.img_min_side = size
        self.img_max_side = 1333

    def get_affine_matrix(self):
        # get affine matrix for transformation
        return ts.random_transform(self.min_rotation, self.max_rotation, self.min_translation, self.max_translation, self.min_shear, self.max_shear, self.min_scaling, self.max_scaling, self.flip_x_chance, self.flip_y_chance, self.prng)

    def __call__(self, img, boxes, labels):

        original_width, original_height = img.shape[:2]
        if len(labels) != 0:
            boxes[:, 0] *= original_width
            boxes[:, 1] *= original_height
            boxes[:, 2] *= original_width
            boxes[:, 3] *= original_height
            boxes = np.rint(boxes)
            boxes = boxes.astype(int)

        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        img = img.astype(np.float32)
        delta = random.randint(-30, 30)
        img += delta
        np.clip(img, 0, 255, out=img)
        img = img.astype(np.float32) / 255.
        img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
        img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
        img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225

        affine_matrix = self.get_affine_matrix()
        transform = ts.adjust_transform_for_image(affine_matrix, img, True)
        img = image.apply_transform(transform, img, image.TransformParameters())

        boxes = boxes.copy()
        for index in range(boxes.shape[0]):
            boxes[index, :] = ts.transform_aabb(transform, boxes[index, :])
        boxes = boxes.astype(np.float32)
        img, img_scale = image.resize_image(img, self.img_min_side, self.img_max_side)
        boxes *= img_scale

        return img, boxes, labels


class RetinaTransformVal:
    def __init__(self, size):
        self.img_min_side = size
        self.img_max_side = 1333

    def __call__(self, img, boxes, labels):
        original_width, original_height = img.shape[:2]
        # convert from relative to absolute coordinate

        if len(labels) != 0:
            boxes[:, 0] *= original_width
            boxes[:, 1] *= original_height
            boxes[:, 2] *= original_width
            boxes[:, 3] *= original_height
            boxes = np.rint(boxes)
            boxes = boxes.astype(int)

        # expand image
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        img = img.astype(np.float32) / 255.
        img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
        img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
        img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225

        boxes = boxes.copy()
        boxes = boxes.astype(np.float32)
        img, img_scale = image.resize_image(img, self.img_min_side, self.img_max_side)
        boxes *= img_scale

        return img, boxes, labels