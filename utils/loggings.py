import os
import datetime
from utils import logger
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter


def check_asset_dir(asset_path, config):
    if not os.path.exists(asset_path):
        os.makedirs(asset_path)
        config.save(os.path.join(asset_path, "hparams.yaml"))
    if not os.path.exists(os.path.join(asset_path, 'model')):
        os.makedirs(os.path.join(asset_path, 'model'))
    if not os.path.exists(os.path.join(asset_path, 'result')):
        os.makedirs(os.path.join(asset_path, 'result', 'sample'))


def print_result(losses, metrics):
    for name, val in losses.items():
        logger.info("%s: %.4f" % (name, val))
    for name, val in metrics.items():
        logger.info("%s: %.4f" % (name, val))
        

def get_tflogger(asset_path):
    now = datetime.datetime.now()
    folder = "run-%s" % now.strftime("%m%d-%H%M%S")
    tf_logger = SummaryWriter(os.path.join(asset_path, 'tensorboard', folder))

    return tf_logger


# def tensorboard_logging_result(tf_logger, step, results, **kwargs):
#     for tag, value in results.items():
#         if 'img' in tag:
#             tf_logger.add_image(tag, value, step)
#         elif 'hist' in tag:
#             tf_logger.add_histogram(tag, value, step)
#         elif 'pr' in tag:
#             tf_logger.add_pr_curve(tag, kwargs['labels'], kwargs['predictions'], step)
#         else:
#             tf_logger.add_scalaar(tag, value, step)
