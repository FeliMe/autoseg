import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from skimage.transform import resize
import torch
import torch.nn.functional as F


def volume_viewer(volume):
    """Plot a volume of shape [x, y, slices]
    Useful for MR and CT image volumes

    Args:
        volume (torch.Tensor or np.ndarray): With shape [slices, h, w]"""
    
    def remove_keymap_conflicts(new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    def previous_slice(ax):
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
        ax.images[0].set_array(volume[ax.index])
        ax.texts.pop()
        ax.text(5, 15, f"Slice: {ax.index}", color="white")

    def next_slice(ax):
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index])
        ax.texts.pop()
        ax.text(5, 15, f"Slice: {ax.index}", color="white")

    def process_key(event):
        fig = event.canvas.figure
        # Move axial (slices)
        if event.key == 'j':
            next_slice(fig.axes[0])
        elif event.key == 'k':
            previous_slice(fig.axes[0])
        # Move coronal (h)
        elif event.key == 'u':
            previous_slice(fig.axes[1])
        elif event.key == 'i':
            next_slice(fig.axes[1])
        # Move saggital (w)
        elif event.key == 'h':
            previous_slice(fig.axes[2])
        elif event.key == 'l':
            next_slice(fig.axes[2])
        fig.canvas.draw()

    def prepare_volume(volume):
        # Convert to torch
        if isinstance(volume, np.ndarray):
            try:
                volume = torch.from_numpy(volume)
            except ValueError:
                volume = torch.from_numpy(volume.copy())

        # Omit batch dimension
        if volume.ndim == 4:
            volume = volume[0]

        # Pad slices
        if volume.shape[0] < volume.shape[1]:
            pad = (volume.shape[1] - volume.shape[0]) // 2
            volume = F.pad(volume, [0, 0, 0, 0, pad, pad])

        # Transform such that axial view is first
        volume = torch.flip(volume, [2, 1, 0])
        volume = volume.permute(2, 1, 0)

        return volume

    def plot_ax(ax, volume, title):
        ax.volume = volume
        shape = ax.volume.shape
        ax.index = shape[0] // 2
        aspect = shape[2] / shape[1]
        ax.imshow(ax.volume[ax.index], aspect=aspect)
        ax.set_title(title)
        ax.text(5, 15, f"Slice: {ax.index}", color="white")

    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['image.interpolation'] = 'nearest'

    remove_keymap_conflicts({'h', 'j', 'k', 'l'})

    volume = prepare_volume(volume)

    # Volume shape [slices, h, w]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    plot_ax(ax[0], volume, "axial")  # axial [slices, h, w]
    plot_ax(ax[1], volume.permute(1, 0, 2), "coronal")  # saggital [h, slices, w]
    plot_ax(ax[2], volume.permute(2, 0, 1), "saggital")  # coronal [w, slices, h]
    fig.canvas.mpl_connect('key_press_event', process_key)
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


def load_nii(path: str, size: int = None, dtype : str = "float32"):
    """Load a neuroimaging file with nibabel, [w, h, slices]
    https://nipy.org/nibabel/reference/nibabel.html

    Args:
        path (str): Path to nii file
        size (int): Optional. Output size for h and w. Only supports rectangles
        dtype (str): Numpy datatype

    Returns:
        volume (np.ndarray): Of shape [w, h, slices]
        affine (np.ndarray): Affine coordinates (rotation and translation),
                             shape [4, 4]
    """
    # Load file
    data = nib.load(path, keep_file_open=False)
    volume = data.get_fdata(caching='unchanged', dtype=np.dtype(dtype))
    affine = data.affine

    # Squeeze optional 4th dimension
    if volume.ndim == 4:
        volume = volume.squeeze(-1)

    # Resize if size is given
    if size is not None:
        volume = resize(volume, [volume.shape[0], size, size])

    return volume, affine


def save_nii(path: str, volume: np.ndarray, affine: np.ndarray, dtype: str = "float32"):
    nib.save(nib.Nifti1Image(volume.astype(dtype), affine), path)


if __name__ == '__main__':
    path = "/home/felix/datasets/MOOD/brain/train/00000.nii.gz"
    volume, affine = load_nii(path)
    volume_viewer(volume)
