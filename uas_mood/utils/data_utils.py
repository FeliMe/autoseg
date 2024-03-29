from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from skimage.exposure import equalize_hist
from skimage.transform import resize
import torch
from torchvision import transforms


def plot(image, f=None):
    plt.axis("off")
    plt.imshow(image, cmap="gray", vmin=0., vmax=1.)
    if f is None:
        plt.show()
    else:
        plt.savefig(f, bbox_inches='tight', pad_inches=0)


def volume_viewer(volume, initial_position=None, slices_first=True):
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
        # Convert to numpy
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()

        # Omit batch dimension
        if volume.ndim == 4:
            volume = volume[0]

        # If image is not loaded with slices_first, put slices dimension first
        if not slices_first:
            volume = np.moveaxis(volume, 2, 0)

        # Pad slices
        if volume.shape[0] < volume.shape[1]:
            pad_size = (volume.shape[1] - volume.shape[0]) // 2
            pad = [(0, 0)] * volume.ndim
            pad[0] = (pad_size, pad_size)
            volume = np.pad(volume, pad)

        # Flip directions for display
        volume = np.flip(volume, (0, 1, 2))

        return volume

    def plot_ax(ax, volume, index, title):
        ax.volume = volume
        shape = ax.volume.shape
        d = shape[0]
        ax.index = d - index
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
    plot_ax(ax[0], np.transpose(volume, (0, 2, 1)), initial_position[2],
            "axial")  # axial [slices, h, w]
    plot_ax(ax[1], np.transpose(volume, (2, 0, 1)), initial_position[1],
            "coronal")  # saggital [h, slices, w]
    plot_ax(ax[2], np.transpose(volume, (1, 0, 2)), initial_position[0],
            "sagittal")  # coronal [w, slices, h]
    fig.canvas.mpl_connect('key_press_event', process_key)
    print("Plotting volume, navigate:"
          "\naxial with 'j', 'k'"
          "\ncoronal with 'u', 'i'"
          "\nsaggital with 'h', 'l'")
    plt.show()


def write_txt(path: str, msg: str) -> None:
    with open(path, "w") as f:
        f.write(msg)


def load_nii(path: str, size: int = None, primary_axis: int = 0,
             dtype: str = "float32"):
    """Load a neuroimaging file with nibabel, [w, h, slices]
    https://nipy.org/nibabel/reference/nibabel.html

    Args:
        path (str): Path to nii file
        size (int): Optional. Output size for h and w. Only supports rectangles
        primary_axis (int): Primary axis (the one to slice along, usually 2)
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
        volume = resize(volume, [size, size, size])

    # Convert
    volume = volume.astype(np.dtype(dtype))

    # Move primary axis to first dimension
    volume = np.moveaxis(volume, primary_axis, 0)

    return volume, affine


def save_nii(path: str, volume: np.ndarray, affine: np.ndarray = None,
             dtype: str = "float32", primary_axis: int = 0) -> None:
    """Save a neuroimaging file (.nii) with nibabel
    https://nipy.org/nibabel/reference/nibabel.html

    Args:
        path (str): Path to save file at
        volume (np.ndarray): Image as numpy array
        affine (np.ndarray): Affine transformation that determines the
                             world-coordinates of the image elements
        dtype (str): Numpy dtype of saved image
        primary_axis (int): The primary axis. Needs to be put back in place
    """
    if affine is None:
        affine = np.eye(4)
    volume = np.moveaxis(volume, 0, primary_axis)
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


def process_scan(path: str, size: int = None, equalize_hist: bool = False,
                 return_affine: bool = False) -> np.ndarray:
    """Load and pre-process a medical 3D scan

    Args:
        path (str): Path to file
        size (int): Optional, spatial dimension (height / width)
        equalize_hist (bool): Perform histogram equalization
        return_affine (bool): Whether to return the affine transformation matrix

    Returns:
        volume (torch.Tensor): Loaded and pre-processed scan
        affine (np.ndarray): Affine transformation matrix
    """

    # Load
    volume, affine = load_nii(path=path, size=size, primary_axis=2, dtype="float32")

    # Pre-processing
    if equalize_hist:
        volume = histogram_equalization(volume)

    if return_affine:
        return volume, affine
    else:
        return volume


def load_segmentation(path: str, size: int = None, bin_threshold: float = 0.4):
    """Load a segmentation file

    Args:
        path (str): Path to file
        size (int): Optional, spatial dimension (height / width)
        bin_threshold (float): Optional, threshold at which a pixel belongs to
                               the segmentation
    """

    # Load
    segmentation, _ = load_nii(path, size=size, primary_axis=2, dtype='float32')

    # Binarize
    segmentation = np.where(
        segmentation > bin_threshold, 1, 0).astype(np.short)

    return segmentation


def load_image(path: str, img_size: int = None):
    img = Image.open(path).convert("L")
    if img_size is None:
        return transforms.ToTensor()(img)
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])(img)


if __name__ == '__main__':
    size = 256
    # path = "/home/felix/datasets/MOOD/brain/test_label/pixel/00480_uniform_shift.nii.gz"
    # path = "/home/felix/datasets/MOOD/abdom/test_label/pixel/00330_slice_shuffle.nii.gz"
    # segmentation = load_segmentation(path, size=size)
    # path = "/home/felix/datasets/MOOD/brain/test/00480_uniform_shift.nii.gz"
    # path = "/home/felix/datasets/MOOD/abdom/test/00330_slice_shuffle.nii.gz"
    path = "/home/felix/datasets/MOOD/brain/train/00000.nii.gz"
    volume = process_scan(path, size=size, equalize_hist=False)
    print(volume.shape)
    volume_viewer(volume)
    import IPython ; IPython.embed() ; exit(1)
