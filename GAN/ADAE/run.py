from utils.loggings import *
from utils.hparams import HParams
from utils.data_loader import get_loader
from utils.io import get_project_root

import argparse
import torch
import numpy as np
import random


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    use_cuda = torch.cuda.is_available()
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    # load configuration from yaml file
    config = HParams.load(os.path.join(os.getcwd(), "hparams.yaml"))
    data_config = config.data_io
    model_config = config.model
    exp_config = config.experiment

    # check asset dir and get logger
    root_dir =  "/" if use_cuda else get_project_root("Deep-Generative-Model")
    asset_path = os.path.join(root_dir, "assets", "test")  # change subdirectory
    check_asset_dir(asset_path, config)
    logger.logging_verbosity(1)
    logger.add_filehandler(os.path.join(asset_path, "log.txt"))
    data_config['root_path'] = os.path.join(root_dir, data_config['root_path'])
    
    # seed
    if args.seed > 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    train_loader = get_loader(train=True, **data_config)

    for batch in train_loader:
        print(batch[0].size())
        print(batch[1])
        break
