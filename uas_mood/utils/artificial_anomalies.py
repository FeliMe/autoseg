import random

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.ndimage import filters
from skimage.draw import ellipse

from uas_mood.utils.data_utils import process_scan


def plot(images):
    if not isinstance(images, list):
        images = [images]
    n = len(images)
    fig = plt.figure(figsize=(4 * n, 4))
    plt.axis("off")
    for i, im in enumerate(images):
        fig.add_subplot(1, n, i + 1)
        plt.imshow(im, cmap="gray", vmin=0., vmax=1.)
    plt.show()


def sample_location(img: np.ndarray, back_val: float):
    obj_inds = np.where(img > back_val)
    if len(obj_inds[0]) == 0:
        center = [dim // 2 for dim in img.shape]
    else:
        location_idx = random.randint(0, len(obj_inds[0]) - 1)
        center = [obj_inds[0][location_idx],
                  obj_inds[1][location_idx]]
    return center


def create_rectangle(center, size, img_size):
    """Create a path of patch_size on an image of size at location
    patch_center.

    :param tuple(int, int) center: Center of the sphere
    :param tuple(int, int) size: width and height of the patch
    :param tuple(int, int) img_size: Size of the created mask
    :return np.ndarray patch_mask: Mask of img_size with the patch inside
    """
    if not isinstance(size, np.ndarray):
        size = np.array(size)
    upper_left = np.round(size / 2).astype(int)
    lower_right = np.round(size - upper_left).astype(int)
    starts = center - upper_left
    ends = center + lower_right
    starts = starts.clip(0, img_size)
    ends = ends.clip(0, img_size)
    slices = tuple(
        slice(start, end)
        for start, end in zip(starts, ends)
    )

    patch_mask = np.zeros(img_size, dtype=np.float32)
    patch_mask[slices] = 1

    return patch_mask


def create_ellipse(center, size, img_size):
    """Create a path of patch_size on an image of size at location
    patch_center.

    :param tuple(int, int) center: Center of the sphere
    :param tuple(int, int) size: width and height of the patch
    :param tuple(int, int) img_size: Size of the created mask
    :return np.ndarray patch_mask: Mask of img_size with the patch inside
    """
    rotation = np.random.randint(0, 360)
    mask = np.zeros(img_size, dtype=np.float32)
    radius = size / 2
    rr, cc = ellipse(*center, *radius, rotation=np.deg2rad(rotation))
    mask[rr, cc] = 1
    return mask


def create_polygon(center, size, img_size, n_vertices, order):
    """Create a random polygon with n_vertices.

    Args:
    :param int n_vertices: Number of vertices
    :param tuple(int, int) img_shape: (width, height)
    :param tuple(int, int) center: center coordinates (x, y)
    :param tuple(int, int) scale: scaling factors (x, y)
    :param int order: Select 1 for streight lines, 3 for cubic splines
    :param int blur_factor: Select 1 for streight lines, 3 for cubic splines
    :param np.ndarray poly_mask: Mask of img_size with the polygon inside
    """
    # Sample random radius
    r = np.random.uniform(0.1, 0.5, n_vertices)

    # Sample random degrees
    d_phi = np.random.uniform(-10, 10, n_vertices)
    phi = np.empty(n_vertices)
    for i in range(n_vertices):
        phi[i] = (i * 360. / n_vertices + d_phi[i]) * np.pi / 180.

    # Create x- and y-coordinates of points from degrees and radius
    x = np.cos(phi) * r
    y = np.sin(phi) * r

    # Create line or spline
    tck, _ = interpolate.splprep([x, y], s=0, k=order)
    unew = np.arange(0, 1.01, 0.01)
    poly_points = interpolate.splev(unew, tck)
    poly_points = np.array(poly_points).T

    # Scale and shift
    poly_points *= size
    poly_points += center

    # Render polygon to image mask
    img = Image.new("L", size=img_size, color=0)
    ImageDraw.Draw(img).polygon(poly_points.flatten().tolist(),
                                outline=1, fill=1)
    poly_mask = np.array(img, dtype=np.float32)

    return poly_mask


def sample_patch(img, size_range, data, patch_type, poly_type, n_vertices):
    """Sample a patch of random size at a random location

    :param np.ndarray img: Original image to create the patch for, shape [w, h]
    :param tuple(int, int) size_range: Minimum and maximum radius as fraction of image size
    :param str data: "brain" or "abdom"
    :param str patch_type: "rectangle", "ellipse" or "polygon"
    :param str poly_type: "linear" or "cubic"
    :param int n_vertices: Only relevant for "polygon"
    :return np.ndarray mask: mask with same size as img with sampled patch
    """
    # Sample location
    back_val = 0. if data == "brain" else 1e-3
    center = sample_location(img, back_val)

    # Sample size
    d = img.shape[-1]
    size = np.random.uniform(
        size_range[0] * d, size_range[1] * d, 2).round()

    # Create mask
    img_size = img.shape
    if patch_type == "rectangle":
        mask = create_rectangle(
            center=center,
            size=size,
            img_size=img_size,
        )
    elif patch_type == "polygon":
        mask = create_polygon(
            center=center,
            size=size,
            img_size=img_size,
            order=1 if poly_type == "linear" else 3,
            n_vertices=n_vertices,
        )
    elif patch_type == "ellipse":
        mask = create_ellipse(
            center=center,
            size=size,
            img_size=img_size,
        )
    else:
        raise NotImplementedError

    return mask


def sample_complete_mask(n_patches, blur_prob, img, **kwargs):
    """Sample a mask with n_patches using the sample_patch function and blur

    :param int n_patches: number of patches in mask
    :param float blur_prob: Probability of blurring applied to mask
    :return np.ndarray mask: mask with multiple patches
    """
    mask = np.zeros_like(img)
    for _ in range(n_patches):
        mask = np.logical_or(mask, sample_patch(img, **kwargs))

    if random.random() < blur_prob:
        mask = filters.gaussian_filter(mask * 255, sigma=2) / 255

    return mask.astype(np.float32)


def patch_exchange(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray):
    """Create a sample where one patch is switched from another sample
    with a random interpolation factor

    Args:
    :param np.ndarray img1: shape [w, h]
    :param np.ndarray img2: shape [w, h]
    :param np.ndarray mask: shape [w, h], mask indicating the pixels to swap
    """
    zero_mask = 1 - mask

    # Sample interpolation factor alpha
    alpha = random.uniform(0.05, 0.95)
    alpha = 0.95

    # Target pixel value is also alpha
    patch = mask * alpha
    patch_inv = mask - patch

    # Interpolate between patches
    patch_set = patch * img2 + patch_inv * img1
    patchex = img1 * zero_mask + patch_set

    valid_label = (
        mask * img1)[..., None] != (mask * img2)[..., None]
    valid_label = np.any(valid_label, axis=-1)
    label = valid_label * patch

    return patchex, label


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    i_slice = 100
    path = "/home/felix/datasets/MOOD/brain/test/00480_reflection.nii.gz"
    volume1 = process_scan(path, size=256, slices_lower_upper=[23, 200])
    img1 = volume1[i_slice]
    path = "/home/felix/datasets/MOOD/brain/test/00481_uniform_addition.nii.gz"
    volume2 = process_scan(path, size=256, slices_lower_upper=[23, 200])
    img2 = volume2[i_slice]
    mask = sample_complete_mask(
        n_patches=1, blur_prob=0., img=img1, size_range=[0.1, 0.4],
        data="brain", patch_type="polygon", poly_type="cubic", n_vertices=10
    )
    patchex, label = patch_exchange(img1, img2, mask)

    img = (img1 + mask).clip(0., 1.)
    plt.imshow(img, cmap="gray")
    plt.show()
    plt.imshow(label, cmap="gray")
    plt.show()
    plt.imshow(patchex, cmap="gray")
    plt.show()
    import IPython
    IPython.embed()
    exit(1)
