import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
import torch

from uas_mood.utils.data_utils import load_nii, volume_viewer


def rand_sign():
    return 1 if np.random.random() < 0.5 else -1


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
    """Create a path of patch_size on an image or volume of size at location
    patch_center.

    Args:
        path_size (int)
        patch_center (list or tuple of length 3): Center of the sphere
        size (list or tuple of length 3): Size of the created volume

    Returns:
        sphere (np.array of shape size): Volume with the sphere inside
    """
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


def uniform_addition_anomaly(volume, mask):
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
    intensity_range = np.max(volume) - np.min(volume)
    intensity = np.random.uniform(0.2 * intensity_range, 0.3 * intensity_range)
    intensity *= rand_sign()

    # Apply anomaly
    volume += intensity * mask
    volume = np.clip(volume, 0., 1.)

    return volume, mask


def noise_addition_anomaly(volume, mask):
    """Adds noise sampled from a normal distribution with mu and std
    to voxels of a volume at mask and only at pixels where the object is

    Args:
        volume (np.array): Scan to be augmented, shape [..., slices, h, w]
        mask (np.array): Indicates where to add the anomaly, shape [slices, h, w]
        mu (float): Mean of intensity
        std (float): Standard deviation of intensity
    """

    # Sample random noise
    intensity_range = np.max(volume) - np.min(volume)
    intensity = np.random.uniform(
        0.05 * intensity_range, 0.3 * intensity_range, size=mask.shape)
    intensity *= rand_sign()

    # Reduce noise to mask only
    sphere_add = mask * intensity

    # Apply noise
    volume = volume + sphere_add

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
    dims = np.array(np.shape(volume))
    core = dims // 2  # width of core region
    offset = core // 2  # sampling range of center
    rng = [slice(o, c + o) for c, o in zip(core, offset)]  # Sampling range

    # Select a center from the nonzero pixels in volume
    if volume.shape[0] == 256:  # Brain
        inds = np.where(volume[rng[0], rng[1], rng[2]] > 0)
    elif volume.shape[0] == 512:  # Abdomen has no 0 intensities
        inds = np.where(volume[rng[0], rng[1], rng[2]] > 5e-2)
    else:
        raise RuntimeError("Invalid volume size")

    if isinstance(inds, torch.Tensor):
        inds = inds.numpy()
    # Convert from tuple to array
    inds = np.array(inds)

    # Add the cropped offset from before
    inds = inds + offset[:, None]

    ix = np.random.choice(len(inds[0]))
    center = [ind[ix] for ind in inds[-3:]]

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
    # anomaly_type = "uniform_shift"

    # Sample random center position inside the anatomy
    center = sample_location(volume)

    # Select a radius at random
    d = volume.shape[0]
    if anomaly_type in ["uniform_addition", "noise_addition"]:
        min_radius = np.round(0.05 * d)
        max_radius = np.round(0.10 * d)
    else:
        min_radius = np.round(0.08 * d)
        max_radius = np.round(0.13 * d)
    radius = np.random.randint(min_radius, max_radius)

    # Create sphere with samples radius and location
    sphere = create_sphere(radius, center, list(volume.shape))

    # Truncate sphere to include only voxels inside the object
    sphere = truncate_mask(volume, sphere)

    if verbose:
        print(f"{anomaly_type} at center {center} with radius {radius}")

    # Create anomaly
    if anomaly_type == "uniform_addition":
        res, segmentation = uniform_addition_anomaly(volume, sphere)
    elif anomaly_type == "noise_addition":
        res, segmentation = noise_addition_anomaly(volume, sphere)
    elif anomaly_type == "sink_deformation":
        res, segmentation = sink_deformation_anomaly(
            volume, sphere, center, radius)
    elif anomaly_type == "source_deformation":
        res, segmentation = source_deformation_anomaly(
            volume, sphere, center, radius)
    elif anomaly_type == "uniform_shift":
        res, segmentation = uniform_shift_anomaly(volume, sphere)
    else:
        res, segmentation = reflection_anomaly(volume, sphere)

    return res, segmentation, anomaly_type, center, radius


if __name__ == "__main__":
    # random seed
    # np.random.seed(1)

    # Load a sample
    path = "/home/felix/datasets/MOOD/brain/train/00000.nii.gz"
    volume, affine = load_nii(path)
    print(volume.min(), volume.max(), volume.mean())

    anomal_sample, segmentation, _, center, _ = create_random_anomaly(
        volume, verbose=True)
    print(anomal_sample.min(), anomal_sample.max(), anomal_sample.mean())
    print("Visualizing")
    volume_viewer(anomal_sample, initial_position=center)
    volume_viewer(segmentation, initial_position=center)
    import IPython
    IPython.embed()
    exit(1)
