import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
import torch

from uas_mood.utils.data_utils import load_nii, volume_viewer


def create_sphere(radius, position, size):
    """Create a shpere with radius on a volume of size at position

    Args:
        radius (int)
        position (list or tuple of length 3): Center of the sphere
        size (list or tuple of length 3): Size of the created volume

    Returns:
        sphere (np.array of shape size): Volume with the sphere inside
    """
    assert len(position) == 3
    assert len(size) == 3

    x, y, z = position
    w, h, c = size
    x_, y_, z_ = np.ogrid[-x:c-x, -y:h-y, -z:w-z]
    mask = x_*x_ + y_*y_ + z_*z_ <= radius*radius
    sphere = np.zeros(size)
    sphere[mask] = 1
    return sphere


def create_patch(patch_size, patch_center, size):
    """Create a path of patch_size on a volume of size at location patch_center

    Args:
        path_size (int)
        patch_center (list or tuple of length 3): Center of the sphere
        size (list or tuple of length 3): Size of the created volume

    Returns:
        sphere (np.array of shape size): Volume with the sphere inside
    """
    assert len(patch_center) == 3
    assert len(size) == 3

    upper_left = np.round(patch_size / 2).astype(int)
    lower_right = np.round(patch_size - upper_left).astype(int)
    starts = patch_center - upper_left
    ends = patch_center + lower_right
    starts = starts.clip(0, size)
    ends = ends.clip(0, size)
    slices = tuple(
        slice(start, end)
        for start, end in zip(starts, ends)
    )

    patch = np.zeros(size, dtype=np.float32)
    patch[slices] = 1

    return patch


def uniform_addition_anomaly(volume, mask, mu=0.0, std=0.3):
    """Adds uniform noise sampled from a normal distribution with mu and std
    to a volume at mask and only at pixels where the object is

    Args:
        volume (np.array): Scan to be augmented, shape [..., slices, h, w]
        mask (np.array): Indicates where to add the anomaly, shape [slices, h, w]
        mu (float): Mean of intensity
        std (float): Standard deviation of intensity
    """

    assert isinstance(volume, np.ndarray) or isinstance(volume, torch.Tensor)

    # Sample intensity
    intensity = np.random.normal(mu, std)

    # Apply anomaly
    volume += intensity * mask
    volume = np.clip(volume, 0., 1.)

    return volume, mask


def noise_addition_anomaly(volume, mask, mu=0., std=0.1):
    """Adds noise sampled from a normal distribution with mu and std
    to voxels of a volume at mask and only at pixels where the object is

    Args:
        volume (np.array): Scan to be augmented, shape [..., slices, h, w]
        mask (np.array): Indicates where to add the anomaly, shape [slices, h, w]
        mu (float): Mean of intensity
        std (float): Standard deviation of intensity
    """

    noise_mask = np.zeros_like(volume)
    intensities = np.random.normal(mu, std, size=np.count_nonzero(mask))
    inds = np.where(mask > 0)
    for i, (x, y, z) in enumerate(zip(*inds[-3:])):
        # noise_mask[..., x, y, z] = intensities[i]
        noise_mask[x, y, z] = intensities[i]

    # volume += noise_mask * obj_mask
    volume += noise_mask
    volume = np.clip(volume, 0., 1.)

    return volume, mask


def sink_deformation_anomaly(volume, mask, center, radius):
    """Voxels are shifted toward from the center of the sphere.

    Args:
        volume (np.array): Scan to be augmented, shape [..., slices, h, w]
        mask (np.array): Indicates where to add the anomaly, shape [slices, h, w]
        center (list of length 3): Center pixel of the mask
    """

    # Center voxel of deformation
    C = np.array(center)

    # Create copy of volume for reference
    copy = volume.copy()

    # Iterate over indices of all voxels in mask
    inds = np.where(mask > 0)
    for x, y, z in zip(*inds[-3:]):
        # Voxel at current location
        I = np.array([x, y, z])

        # Sink pixel shift
        s = np.square(np.linalg.norm(I - C, ord=2) / radius)
        V = np.round(I + (1 - s) * (I - C)).astype(np.int)
        x_, y_, z_ = V

        # Assure that z_, y_ and x_ are valid indices
        x_ = max(min(x_, volume.shape[-3] - 1), 0)
        y_ = max(min(y_, volume.shape[-2] - 1), 0)
        z_ = max(min(z_, volume.shape[-1] - 1), 0)

        if volume[..., x, y, z] > 0:
            volume[..., x, y, z] = copy[..., x_, y_, z_]

    return volume, mask


def source_deformation_anomaly(volume, mask, center, radius):
    """Voxels are shifted away from the center of the sphere.

    Args:
        volume (np.array): Scan to be augmented, shape [..., slices, h, w]
        mask (np.array): Indicates where to add the anomaly, shape [slices, h, w]
        center (list of length 3): Center pixel of the mask
    """

    # Center voxel of deformation
    C = np.array(center)

    # Create copy of volume for reference
    copy = volume.copy()

    # Iterate over indices of all voxels in mask
    inds = np.where(mask > 0)
    for x, y, z in zip(*inds[-3:]):
        # Voxel at current location
        I = np.array([x, y, z])

        # Source pixel shift
        s = np.square(np.linalg.norm(I - C, ord=2) / radius)
        V = np.round(C + s * (I - C)).astype(np.int)
        x_, y_, z_ = V

        # Assure that z_, y_ and x_ are valid indices
        x_ = max(min(x_, volume.shape[-1] - 1), 0)
        y_ = max(min(y_, volume.shape[-2] - 1), 0)
        z_ = max(min(z_, volume.shape[-3] - 1), 0)

        if volume[..., x, y, z] > 0:
            volume[..., x, y, z] = copy[..., x_, y_, z_]

    return volume, mask


def uniform_shift_anomaly(volume, mask):
    """Voxels in the sphere are resampled from a copy of the volume which has
    been shifted by a random distance in a random direction.

    Args:
        volume (np.array): Scan to be augmented, shape [..., slices, h, w]
        mask (np.array): Indicates where to add the anomaly, shape [slices, h, w]
    """

    def shift_volume(volume, x, y, z):
        """Shifts a volume by x, y, and z. Only small shifts are supported"""
        shifted = np.roll(volume, shift=x, axis=-1)
        shifted = np.roll(shifted, shift=y, axis=-2)
        shifted = np.roll(shifted, shift=z, axis=-3)
        return shifted

    def rand_sign():
        return 1 if np.random.random() < 0.5 else -1

    # Create shift parameters
    c, h, w = volume.shape[-3:]
    x = rand_sign() * np.random.randint(0.02 * w, 0.05 * w)
    y = rand_sign() * np.random.randint(0.02 * h, 0.05 * h)
    z = rand_sign() * np.random.randint(0.02 * c, 0.05 * c)

    # Shift volume by x, y, z
    shifted = shift_volume(volume, x, y, z)

    # Create anomaly at mask
    volume[mask > 0] = shifted[mask > 0]

    return volume, mask


def reflection_anomaly(volume, mask):
    """pixels in the sphere are resampled from a copy of the volume that has
    been reflected along an axis of symmetry

    Args:
        volume (np.array): Scan to be augmented, shape [..., slices, h, w]
        mask (np.array): Indicates where to add the anomaly, shape [slices, h, w]
    """

    # Create a reflection by flipping along the width axis
    reflection = np.flip(volume, axis=0)

    # Create anomaly at mask
    volume[mask > 0] = reflection[mask > 0]

    return volume, mask


def sample_location(volume):
    # Select a center from the nonzero pixels in volume
    if volume.shape[0] == 256:  # Brain
        nonzero_inds = np.where(volume > 0)
    elif volume.shape[0] == 512:  # Abdomen has no 0 intensities
        nonzero_inds = np.where(volume > 5e-2)
    else:
        raise RuntimeError("Invalid volume size")

    if isinstance(nonzero_inds, torch.Tensor):
        nonzero_inds = nonzero_inds.numpy()
    ix = np.random.choice(len(nonzero_inds[0]))
    center = [ind[ix] for ind in nonzero_inds[-3:]]

    return center


def truncate_mask(volume, mask):
    """Returns a mask only where the object in volume and mask overlay

    Args:
        volume (np.ndarray): 3D scan
        mask (np.ndarray): binary mask (sphere or patch)
    Returns:
        mask (np.ndarray): binary mask at object voxels
    """
    if volume.shape[0] == 256:  # Brain, use nonzero pixels
        obj_mask = np.where(volume > 0, 1, 0)
    elif volume.shape[0] == 512:  # Abdomen, use morphological closing
        obj_mask = binary_fill_holes(np.where(volume > 5e-2, 1, 0),
                                     structure=np.ones((2, 2, 2)))
    else:
        raise RuntimeError("Invalid shape. Not brain (256) or abdomen (512)")

    mask = mask * obj_mask
    return mask


def create_random_anomaly(volume, verbose=False):
    assert volume.ndim == 3
    # Sample random center position inside the anatomy
    center = sample_location(volume)

    # Select a radius at random
    d = volume.shape[0] / 2
    radius = np.random.randint(0.1 * d, 0.4 * d)

    # Create sphere with samples radius and location
    sphere = create_sphere(radius, center, list(volume.shape))

    # Truncate sphere to include only voxels inside the object
    sphere = truncate_mask(volume, sphere)

    # Select a random anomaly
    anomalies = [
        "uniform_addition",
        "noise_addition",
        "sink_deformation",
        "source_deformation",
        "uniform_shift",
        "reflection"
    ]
    anomaly_type = np.random.choice(anomalies)

    if verbose:
        print(f"{anomaly_type} at center {center} with radius {radius}")

    # Create anomaly
    if anomaly_type == "uniform_addition":
        res, segmentation = uniform_addition_anomaly(volume, sphere, mu=0.0, std=0.4)
    elif anomaly_type == "noise_addition":
        res, segmentation = noise_addition_anomaly(volume, sphere, mu=0.0, std=0.2)
    elif anomaly_type == "sink_deformation":
        res, segmentation = sink_deformation_anomaly(volume, sphere, center, radius)
    elif anomaly_type == "source_deformation":
        res, segmentation = source_deformation_anomaly(volume, sphere, center, radius)
    elif anomaly_type == "uniform_shift":
        res, segmentation = uniform_shift_anomaly(volume, sphere)
    else:
        res, segmentation = reflection_anomaly(volume, sphere)

    return res, segmentation, anomaly_type, center, radius


if __name__ == "__main__":
    # random seed
    # np.random.seed(1)

    # Load a sample
    path = "/home/felix/datasets/MOOD/abdom/train/00002.nii.gz"
    volume, affine = load_nii(path)
    print(volume.min(), volume.max(), volume.mean())

    anomal_sample, segmentation, _, center, _ = create_random_anomaly(volume, verbose=True)
    print(anomal_sample.min(), anomal_sample.max(), anomal_sample.mean())
    print("Visualizing")
    volume_viewer(anomal_sample, initial_position=center)
    volume_viewer(segmentation, initial_position=center)
    import IPython ; IPython.embed() ; exit(1)
