import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class VolumeViewer:
    def __init__(self):
        """Plot a volume of shape [x, y, slices]
        Useful for MR and CT image volumes"""
        plt.rcParams['image.cmap'] = 'gray'
        plt.rcParams['image.interpolation'] = 'nearest'

        self.remove_keymap_conflicts({'h', 'j', 'k', 'l'})

    @staticmethod
    def remove_keymap_conflicts(new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    def process_key(self, event):
        fig = event.canvas.figure
        # Move axial (slices)
        if event.key == 'j':
            self.next_slice(fig.axes[0])
        elif event.key == 'k':
            self.previous_slice(fig.axes[0])
        # Move coronal (h)
        elif event.key == 'u':
            self.previous_slice(fig.axes[1])
        elif event.key == 'i':
            self.next_slice(fig.axes[1])
        # Move saggital (w)
        elif event.key == 'h':
            self.previous_slice(fig.axes[2])
        elif event.key == 'l':
            self.next_slice(fig.axes[2])
        fig.canvas.draw()

    @staticmethod
    def previous_slice(ax):
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
        ax.images[0].set_array(volume[ax.index])

    @staticmethod
    def next_slice(ax):
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index])

    @staticmethod
    def prepare_volume(volume):
        # Prepare volume
        if isinstance(volume, np.ndarray):
            volume = torch.from_numpy(volume)
        if volume.ndim == 4:
            volume = volume[0]
        if volume.shape[0] < volume.shape[1]:
            # Pad slices
            pad = (volume.shape[1] - volume.shape[0]) // 2
            volume = F.pad(volume, [0, 0, 0, 0, pad, pad])

        return volume

    def plot(self, volume):
        """volume is a torch.Tensor or np.array with shape [slices, h, w]
        and axial viewing direction"""

        def plot_ax(ax, volume, title):
            ax.volume = volume
            shape = ax.volume.shape
            ax.index = shape[0] // 2
            aspect = shape[2] / shape[1]
            ax.imshow(ax.volume[ax.index], aspect=aspect)
            ax.set_title(title)

        volume = self.prepare_volume(volume)

        # Volume shape [slices, h, w]
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        plot_ax(ax[0], volume, "axial")  # axial [slices, h, w]
        plot_ax(ax[1], volume.permute(1, 0, 2), "coronal")  # saggital [h, slices, w]
        plot_ax(ax[2], volume.permute(2, 0, 1), "saggital")  # coronal [w, slices, h]
        fig.canvas.mpl_connect('key_press_event', self.process_key)
        print("Plotting volume, navigate:" \
              "\naxial with 'j', 'k'" \
              "\ncoronal with 'u', 'i'" \
              "\nsaggital with 'h', 'l'")
        plt.show()


class ResizeGray:
    def __init__(self, size, mode='nearest', align_corners=None):
        """Resample a tensor of shape [c, slices, h, w], or [c, h, w] to size
        Arguments are the same as in torch.nn.functional.interpolate, but we
        don't need a batch- or channel dimension here.
        The datatype can only be preserved when using nearest neighbor.

        Example:
        volume = torch.randn(1, 189, 197, 197)
        out = ResizeGray()(volume, size=[189, 120, 120])
        out.shape = [1, 189, 120, 120]
        out.dtype = volume.dtype if mode == 'nearest' else torch.float32
        """
        if isinstance(size, int):
            size = [size, size]
        assert len(size) == 2

        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def __call__(self, volume):
        dtype = volume.dtype
        out = F.interpolate(volume[None].float(),
                            size=[volume.shape[-3]] + self.size,
                            mode=self.mode,
                            align_corners=self.align_corners)[0]
        if self.mode == 'nearest':
            out = out.type(dtype)
        return out


def nii_loader(path: str, size: int = None, view : str = "axial"):
    """Load a neuroimaging file with nibabel
    https://nipy.org/nibabel/reference/nibabel.html

    Args:
        path (str): Path to nii file
        size (int): Optional. Output size for h and w. Only supports rectangles
        view (str): Optional. One of "axial", "coronal", or "saggital"

    Returns:
        volume (torch.Tensor): Of shape [1, slices, h, w]
    """
    # Load file
    data = nib.load(path, keep_file_open=False)
    volume = data.get_fdata(caching='unchanged', dtype=np.float32)

    # Squeeze optional 4th dimension
    if volume.ndim == 4:
        volume = volume.squeeze(-1)

    # Convert to tensor
    volume = torch.from_numpy(volume)

    # Flip directions
    volume = torch.flip(volume, [2, 1, 0])

    # Select viewing direction
    if view == "axial":
        volume = volume.permute(2, 1, 0)
    elif view == "coronal":
        volume = volume.permute(1, 2, 0)
    elif view == "saggital":
        volume = volume.transpose(1, 2)
    else:
        raise NotImplementedError

    volume = volume.unsqueeze(0)

    # Resize if size is given
    if size is not None:
        volume = ResizeGray(size)(volume)

    return volume


if __name__ == '__main__':
    path = "/home/felix/datasets/MOOD/brain/train/00000.nii.gz"
    volume = nii_loader(path)

    VolumeViewer().plot(volume)
