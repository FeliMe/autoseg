from datetime import datetime
import math
from numbers import Number

import numpy as np
from skimage import measure
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def connected_components_3d(volume):
    is_batch = True
    is_torch = torch.is_tensor(volume)
    if volume.ndim == 3:
        volume = volume.unsqueeze(0)
        is_batch = False
    if is_torch:
        volume = volume.numpy()

    # shape [b, slices, w, h], treat every sample in batch independently
    # for i in tqdm(range(len(volume)), desc="Connected components"):
    for i in range(len(volume)):
        cc_volume = measure.label(volume[i], connectivity=3)
        props = measure.regionprops(cc_volume)
        for prop in props:
            if prop['filled_area'] <= 20:
                volume[i, cc_volume == prop['label']] = 0

    if is_torch:
        volume = torch.from_numpy(volume)
    if not is_batch:
        volume = volume.squeeze(0)
    return volume


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super().__init__()
        if isinstance(kernel_size, Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([
            torch.arange(size, dtype=torch.float32) for size in kernel_size
        ])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        self.padding = [k // 2 for k in kernel_size]
        self.padding *= 2

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(
                    dim)
            )

    def forward(self, inp):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        inp = F.pad(inp, self.padding, mode='replicate')
        return self.conv(inp, weight=self.weight, groups=self.groups)
