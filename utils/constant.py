NORMALIZE_FACTOR = {
    "cifar10": [(0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616)],
    "mnist": [(0.1307,), (0.3081,)],
    "imagenet": [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
    "celeba": [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)],
    "pneumonia": [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
}
CROP_SIZE = 178
IMAGE_SIZE = 128