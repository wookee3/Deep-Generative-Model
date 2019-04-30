import os
import datetime
from utils import logger
from .tf_logger import TF_Logger


def check_asset_dir(asset_path, config):
    if not os.path.exists(asset_path):
        os.makedirs(asset_path)
        config.save(os.path.join(asset_path, "hparams.yaml"))
    if not os.path.exists(os.path.join(asset_path, 'model')):
        os.makedirs(os.path.join(asset_path, 'model'))
    if not os.path.exists(os.path.join(asset_path, 'result')):
        os.makedirs(os.path.join(asset_path, 'result', 'sample'))


def get_tflogger(asset_path):
    now = datetime.datetime.now()
    folder = "run-%s" % now.strftime("%m%d-%H%M%S")
    tf_logger = TF_Logger(os.path.join(asset_path, 'tensorboard', folder))

    return tf_logger


def print_result(losses, metrics):
    for name, val in losses.items():
        logger.info("%s: %.4f" % (name, val))
    for name, val in metrics.items():
        logger.info("%s: %.4f" % (name, val))


def tensorboard_logging_result(tf_logger, epoch, results):
    for tag, value in results.items():
        if 'img' in tag:
            tf_logger.image_summary(tag, value, epoch)
        elif 'hist' in tag:
            tf_logger.histo_summary(tag, value, epoch)
        else:
            tf_logger.scalar_summary(tag, value, epoch)
