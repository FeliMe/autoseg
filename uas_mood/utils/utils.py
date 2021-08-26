from datetime import datetime
import math
from numbers import Number
import os
import warnings

import numpy as np
import psutil
from skimage import measure
import torch
import torch.nn as nn
import torch.nn.functional as F


def write_file(path: str, msg: str):
    with open(path, "w") as f:
        f.write(msg)


def read_file(path: str):
    with open(path, 'r') as f:
        data = f.read().replace('\n', '')
    return data


def filelist_size(files):
    """Returns the size of all elements in a list of files"""
    return sum(os.path.getsize(f) for f in files)


def check_ram(files):
    ds_size = filelist_size(files)
    free_ram = psutil.virtual_memory()[4]
    if ds_size > free_ram:
        warnings.warn("Your dataset is larger than your available RAM.")


def printer(msg: str, verbose: bool):
    if verbose:
        print(msg)


def torch2np_img(img):
    """
    Converts a pytorch image into a cv2 RGB image

    Args:
        img (torch.tensor): range (-1, 1), dtype torch.float32, shape (C, H, W)

    Returns:
        img (np.array): range(0, 255), dtype np.uint8, shape (H, W, C)
    """
    return (img.permute(1, 2, 0).numpy() * 255.).astype(np.uint8)


def get_training_timings(start_time, current_epoch, max_epochs):
    time_elapsed = datetime.now() - datetime.fromtimestamp(start_time)
    # self.current_epoch starts at 0
    time_per_epoch = time_elapsed / (current_epoch + 1)
    time_left = (max_epochs - current_epoch - 1) * time_per_epoch
    return time_elapsed, time_per_epoch, time_left
