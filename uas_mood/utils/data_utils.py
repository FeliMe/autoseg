import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from skimage.exposure import equalize_hist
from skimage.transform import resize
import torch
import torch.nn.functional as F


def volume_viewer(volume, initial_position=None, slices_first=False):
    """Plot a volume of shape [x, y, slices]
    Useful for MR and CT image volumes

    Args:
        volume (torch.Tensor or np.ndarray): With shape [slices, h, w]
        initial_position (list or tuple of len 3): (Optional)
        slices_first (bool): If slices are first or last dimension in volume
    """

    def remove_keymap_conflicts(new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    def previous_slice(ax):
        volume = ax.volume
        d = volume.shape[0]
        ax.index = (ax.index + 1) % d
        ax.images[0].set_array(volume[ax.index])
        ax.texts.pop()
        ax.text(5, 15, f"Slice: {d - ax.index}", color="white")

    def next_slice(ax):
        volume = ax.volume
        d = volume.shape[0]
        ax.index = (ax.index - 1) % d
        ax.images[0].set_array(volume[ax.index])
        ax.texts.pop()
        ax.text(5, 15, f"Slice: {d - ax.index}", color="white")

    def process_key(event):
        fig = event.canvas.figure
        # Move axial (slices)
        if event.key == 'k':
            next_slice(fig.axes[0])
        elif event.key == 'j':
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

    def prepare_volume(volume, slices_first):
        # Convert to torch
        if isinstance(volume, np.ndarray):
            try:
                volume = torch.from_numpy(volume)
            except ValueError:
                volume = torch.from_numpy(volume.copy())

        # Omit batch dimension
        if volume.ndim == 4:
            volume = volume[0]

        # If first dimension is slices, put it last
        if slices_first:
            volume = volume.permute(1, 2, 0)

        # Pad slices
        if volume.shape[0] < volume.shape[1]:
            pad = (volume.shape[1] - volume.shape[0]) // 2
            volume = F.pad(volume, [0, 0, 0, 0, pad, pad])

        # Transform such that axial view is first
        volume = torch.flip(volume, [2, 1, 0])
        volume = volume.permute(2, 1, 0)

        return volume

    def plot_ax(ax, volume, index, title):
        ax.volume = volume
        shape = ax.volume.shape
        d = shape[0]
        ax.index = d - index
        # ax.index = index
        aspect = shape[2] / shape[1]
        ax.imshow(ax.volume[ax.index], aspect=aspect, vmin=0., vmax=1.)
        ax.set_title(title)
        ax.text(5, 15, f"Slice: {d - ax.index}", color="white")

    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['image.interpolation'] = 'nearest'

    remove_keymap_conflicts({'h', 'j', 'k', 'l'})

    volume = prepare_volume(volume, slices_first)

    if initial_position is None:
        initial_position = torch.tensor(volume.shape) // 2

    # Volume shape [slices, h, w]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    plot_ax(ax[0], volume, initial_position[2],
            "axial")  # axial [slices, h, w]
    plot_ax(ax[1], volume.permute(1, 0, 2), initial_position[1],
            "coronal")  # saggital [h, slices, w]
    plot_ax(ax[2], volume.permute(2, 0, 1), initial_position[0],
            "saggital")  # coronal [w, slices, h]
    fig.canvas.mpl_connect('key_press_event', process_key)
    print("Plotting volume, navigate:"
          "\naxial with 'j', 'k'"
          "\ncoronal with 'u', 'i'"
          "\nsaggital with 'h', 'l'")
    plt.show()


def load_nii(path: str, size: int = None, dtype: str = "float32"):
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
    volume = data.get_fdata(caching='unchanged')  # [w, h, slices]
    affine = data.affine

    # Squeeze optional 4th dimension
    if volume.ndim == 4:
        volume = volume.squeeze(-1)

    # Resize if size is given and if necessary
    if size is not None and (volume.shape[0] != size or volume.shape[1] != size):
        order = 1 if "float" in dtype else 0
        # order = 0
        volume = resize(volume, [size, size, volume.shape[-1]], order=order)

    volume = volume.astype(np.dtype(dtype))

    return volume, affine


def save_nii(path: str, volume: np.ndarray, affine: np.ndarray, dtype: str = "float32"):
    nib.save(nib.Nifti1Image(volume.astype(dtype), affine), path)


def histogram_equalization(volume):
    # Create equalization mask
    mask = np.zeros_like(volume)
    mask[volume > 0] = 1

    # Equalize
    dtype = volume.dtype
    volume = equalize_hist(volume, nbins=256, mask=mask).astype(dtype)

    # Assure that background still is 0
    volume *= mask

    return volume


def process_scan(path: str, size: int = None, equalize_hist: bool = True,
                 slices_lower_upper=None):
    """Load and pre-process a medical 3D scan

    Args:
        path (str): Path to file
        size (int): Optional, spatial dimension (height / width)
        equalize_hist (bool): Perform histogram equalization
        slices_lower_upper (tuple or list of ints and length 2):
            upper and lower index for slices

    Returns:
        volume (torch.Tensor): Loaded and pre-processed scan
    """

    # Load
    volume, _ = load_nii(path=path, size=size, dtype="float32")

    # Select slices
    if slices_lower_upper is not None:
        volume = volume[..., slice(*slices_lower_upper)]

    # Pre-processing
    if equalize_hist:
        # volume = histogram_equalization(volume)
        pass  # histogram equalization did not help here

    # convert from [w, h, slices] to [slices, w, h]
    # primary axis will be put first, 2 for brain, 1 for abdomen
    primary_axis = 2
    volume = np.rollaxis(volume, primary_axis)

    return volume


def load_segmentation(path: str, size: int = None, bin_threshold: float = 0.4,
                      slices_lower_upper=None):
    """Load a segmentation file

    Args:
        path (str): Path to file
        size (int): Optional, spatial dimension (height / width)
        bin_threshold (float): Optional, threshold at which a pixel belongs to
                               the segmentation
    """

    # Load
    segmentation, _ = load_nii(path, size=size, dtype='float32')

    # Select slices
    if slices_lower_upper is not None:
        segmentation = segmentation[..., slice(*slices_lower_upper)]

    # Binarize
    segmentation = np.where(
        segmentation > bin_threshold, 1, 0).astype(np.short)

    # convert from [w, h, slices] to [slices, w, h]
    # primary axis will be put first, 2 for brain, 1 for abdomen
    primary_axis = 2
    segmentation = np.rollaxis(segmentation, primary_axis)

    return segmentation


if __name__ == '__main__':
    path = "/home/felix/datasets/MOOD/brain/test_label/pixel/00480_reflection.nii.gz"
    segmentation = load_segmentation(
        path, slices_lower_upper=[23, 200], size=128)
    path = "/home/felix/datasets/MOOD/brain/test/00480_reflection.nii.gz"
    volume = process_scan(path, size=128, slices_lower_upper=[23, 200])
    print(volume.shape)
    volume_viewer(volume, slices_first=True)
