import torch
import numpy as np
from uas_mood.utils.data_utils import nii_loader, volume_viewer


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


def uniform_addition_anomaly(volume, mask, mu=0., std=0.1):
    """Adds uniform noise sampled from a normal distribution with mu and std
    to a volume at mask and only at pixels where the object is

    Args:
        volume (np.array): Scan to be augmented, shape [..., slices, h, w]
        mask (np.array): Indicates where to add the anomaly, shape [slices, h, w]
        mu (float): Mean of intensity
        std (float): Standard deviation of intensity
    """

    assert isinstance(volume, np.ndarray) or isinstance(volume, torch.Tensor)

    intensity = np.random.normal(mu, std)
    obj_mask = np.zeros_like(volume)
    obj_mask[volume > 0] = 1  # TODO: Change to convex hull

    # Apply anomaly
    volume += intensity * mask * obj_mask
    volume = np.clip(volume, 0., 1.)

    return volume


def noise_addition_anomaly(volume, mask, mu=0., std=0.1):
    """Adds noise sampled from a normal distribution with mu and std
    to voxels of a volume at mask and only at pixels where the object is

    Args:
        volume (np.array): Scan to be augmented, shape [..., slices, h, w]
        mask (np.array): Indicates where to add the anomaly, shape [slices, h, w]
        mu (float): Mean of intensity
        std (float): Standard deviation of intensity
    """

    assert isinstance(volume, np.ndarray) or isinstance(volume, torch.Tensor)

    obj_mask = np.zeros_like(volume)
    obj_mask[volume > 0] = 1  # TODO: Change to convex hull

    noise_mask = np.zeros_like(volume)
    intensities = np.random.normal(mu, std, size=np.count_nonzero(mask))
    inds = np.where(mask > 0)
    for i, (x, y, z) in enumerate(zip(*inds[-3:])):
        # noise_mask[..., x, y, z] = intensities[i]
        noise_mask[x, y, z] = intensities[i]

    volume += noise_mask * obj_mask
    volume = np.clip(volume, 0., 1.)

    return volume


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
    for z, y, x in zip(*inds[-3:]):
        # Voxel at current location
        I = np.array([z, y, x])

        # Sink pixel shift
        s = np.square(np.linalg.norm(I - C, ord=2) / radius)
        V = np.round(I + (1 - s) * (I - C)).astype(np.int)
        z_, y_, x_ = V
        if volume[..., z, y, x] > 0:
            volume[..., z, y, x] = copy[..., z_, y_, x_]

    return volume


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
    for z, y, x in zip(*inds[-3:]):
        # Voxel at current location
        I = np.array([z, y, x])

        # Source pixel shift
        s = np.square(np.linalg.norm(I - C, ord=2) / radius)
        V = np.round(C + s * (I - C)).astype(np.int)
        z_, y_, x_ = V
        if volume[..., z, y, x] > 0:
            volume[..., z, y, x] = copy[..., z_, y_, x_]

    return volume


def uniform_shift_anomaly(volume, mask):
    """Voxels in the sphere are resampled from a copy of the volume which has
    been shifted by a random distance in a random direction.

    Args:
        volume (np.array): Scan to be augmented, shape [..., slices, h, w]
        mask (np.array): Indicates where to add the anomaly, shape [slices, h, w]
    """

    assert isinstance(volume, np.ndarray) or isinstance(volume, torch.Tensor)

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

    return volume


def reflection_anomaly(volume, mask):
    """pixels in the sphere are resampled from a copy of the volume that has
    been reflected along an axis of symmetry

    Args:
        volume (np.array): Scan to be augmented, shape [..., slices, h, w]
        mask (np.array): Indicates where to add the anomaly, shape [slices, h, w]
    """
    # Create a reflection by flipping along the width axis
    reflection = np.flip(volume, axis=-1)
    # Create anomaly at mask
    volume[mask > 0] = reflection[mask > 0]
    return volume


def sample_location(volume):
    # Select a center from the nonzero pixels in volume
    nonzero_inds = np.nonzero(volume)
    if isinstance(nonzero_inds, torch.Tensor):
        nonzero_inds = nonzero_inds.numpy()
    ix = np.random.choice(len(nonzero_inds[0]))
    center = [ind[ix] for ind in nonzero_inds[-3:]]
    return center


def create_random_anomaly(volume, verbose=False):
    assert volume.ndim == 3
    # Sample random center position inside the anatomy
    center = sample_location(volume)
    center = [128, 150, 150]

    # Select a radius at random
    radius = np.random.randint(5, 40)
    radius = 40

    # Create sphere with samples radius and location
    sphere = create_sphere(radius, center, list(volume.shape))

    # Select a random anomaly
    anomalies = [
        "uniform_addition",
        "noise_addition",
        "sink_deformation",
        "source_deformation",
        "uniform_shift",
        "reflection"
    ]
    anomaly = np.random.choice(anomalies)
    anomaly = "uniform_addition"

    if verbose:
        print(f"{anomaly} at center {center} with radius {radius}")

    # Create anomaly
    if anomaly == "uniform_addition":
        res = uniform_addition_anomaly(volume, sphere, mu=0.0, std=0.1)
    elif anomaly == "noise_addition":
        res = noise_addition_anomaly(volume, sphere, mu=0.0, std=0.1)
    elif anomaly == "sink_deformation":
        res = sink_deformation_anomaly(volume, sphere, center, radius)
    elif anomaly == "source_deformation":
        res = source_deformation_anomaly(volume, sphere, center, radius)
    elif anomaly == "uniform_shift":
        res = uniform_shift_anomaly(volume, sphere)
    else:
        res = reflection_anomaly(volume, sphere)

    return res, anomaly


if __name__ == "__main__":
    # random seed
    np.random.seed(1)

    # Load a sample
    path = "/home/felix/datasets/MOOD/brain/train/00000.nii.gz"
    volume = nii_loader(path)

    anomal_sample, _ = create_random_anomaly(volume, verbose=True)
    volume_viewer(anomal_sample)
    import IPython ; IPython.embed() ; exit(1)
