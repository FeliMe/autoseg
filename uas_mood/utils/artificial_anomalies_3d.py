import os
import random
from shutil import rmtree
from time import time

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy import ndimage
import torch
import torch.nn.functional as F
import trimesh

from uas_mood.utils.data_utils import process_scan, volume_viewer


def plot_points_3d(points: np.ndarray, normals: np.ndarray = None,
                   f: str = None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])

    if normals is not None:
        for p, n in zip(points, normals):
            ax.plot(
                [p[0], p[0] + n[0]],
                [p[1], p[1] + n[1]],
                [p[2], p[2] + n[2]],
                color="red"
            )
    if f is None:
        plt.show()
    else:
        plt.savefig(f)


def plot_volume(volume: np.ndarray, dirname: str):
    if os.path.isdir(dirname):
        rmtree(dirname)
    os.makedirs(dirname)
    for i_slice, img in enumerate(volume):
        plt.figure()
        plt.imshow(img, cmap="gray", vmin=0., vmax=1.)
        plt.savefig(os.path.join(dirname, f"slice_{i_slice}.png"))
        plt.close()


def sample_location_brain_3d(volume: np.ndarray):
    """Sample a random location in a 3D brain mri only at brain voxels"""
    obj_inds = np.where(volume > 0)
    if len(obj_inds[0]) == 0:
        center = [dim // 2 for dim in volume.shape]
    else:
        location_idx = random.randint(0, len(obj_inds[0]) - 1)
        center = [obj_inds[0][location_idx],
                  obj_inds[1][location_idx],
                  obj_inds[2][location_idx]]
    return center


def sample_location_abdom_3d(volume: np.ndarray):
    core_percent = 0.8
    dims = np.array(np.shape(volume))
    core = core_percent * dims
    offset = (1 - core_percent) * dims / 2
    center = [
        np.random.randint(offset[0], offset[0] + core[0]),
        np.random.randint(offset[1], offset[1] + core[1]),
        np.random.randint(offset[2], offset[2] + core[2]),
    ]
    return center


def sample_polygon_points_3d(n_vertices: int, normalize: bool = True) -> tuple:
    """Sample points from a unit circle to create a 3D polygon with n_vertices
    uniformly sampled.

    Args:
        n_vertices (int): number of vertices

    Returns:
        points (np.ndarray): Coordinates of points, shape [n_vertices, xyz]
        normals (np.ndarray): Normal vectors of points, shape [n_vertices, xyz]
    """
    # Sample random radius for each point (radius 0.5 == diameter 1.0)
    r = np.random.uniform(0.1, 0.5, n_vertices)

    # Sample random azimuth (theta) and polar angle (phi) for each point
    theta = np.random.uniform(0, 2 * np.pi, n_vertices)
    phi = np.random.uniform(0, np.pi, n_vertices)

    # Get x, y, z from random radius and angles
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    points = np.stack((x, y, z), axis=1)

    # Get normalized normal vectors (r equals magnitude)
    normals = points / r[:, None]

    # Normalize points to range [-0.5, 0.5]
    if normalize:
        points = points / (points.max() - points.min())

    return points, normals


def transform_trimesh(mesh: trimesh.Trimesh, scale: list, shift: list):
    """Apply an inplace scale and shift to a trimesh"""
    scale_mat = np.diag(scale + [1])
    shift_mat = np.array(12 * [0] + shift + [0]).reshape(4, 4).T
    t = scale_mat + shift_mat
    mesh.apply_transform(t)
    return mesh


def create_mesh(points: np.ndarray, normals: np.ndarray = None):
    """Create a trimesh from a point cloud using the ball pivoting algorithm.

    Args:
        points (np.ndarray): Points of point cloud, shape [n, 3]
        normals (np.ndarray): Normal vectors, shape [n, 3]

    Returns:
        mesh (tirmesh.Trimesh): Watertight mesh from points
    """
    # Create open3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is None:
        pcd.estimate_normals()
    else:
        pcd.normals = o3d.utility.Vector3dVector(normals)

    # Fit mesh with the Ball-Pivoting Algorithm
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,o3d.utility.DoubleVector([radius, radius * 2]))

    # Convert open3d to trimesh
    mesh = trimesh.Trimesh(np.asarray(mesh.vertices),
                           np.asarray(mesh.triangles),
                           vertex_normals=np.asarray(mesh.vertex_normals))

    # Fill holes in mesh
    # trimesh.repair.fill_holes(mesh)
    if not mesh.fill_holes():
        # warn("Mesh is not watertight")
        mesh = trimesh.convex.convex_hull(mesh)

    return mesh


def volume_from_voxels(voxel_grid: trimesh.voxel.VoxelGrid, size: tuple) -> np.ndarray:
    """Creates a 3D volume from a list of voxel indices

    Args:
        voxel_grid (trimesh.voxel.VoxelGrid):
        size (3-tuple): Shape of the resulting 3D volume

    Returns:
        volume (np.ndarray)
    """
    volume = np.zeros(size, dtype=np.float32)
    points = voxel_grid.points.astype(np.int32)
    for dim, limit in enumerate(volume.shape):
        points[:, dim] = points[:, dim].clip(0, limit - 1)
    for point in points:
        volume[tuple(point)] = 1

    return volume


def sample_polygon_3d(volume: np.ndarray, size_range: list, data: str,
                      n_vertices: int):
    """Sample a polygon anomaly mask for a 3d volume.

    Args:
        volume (np.ndarray): 3D volume to sample the mask for
        size_range (list): Lower and upper possible size (len = 2)
        data (str): "brain" or "abdom"
        n_vertices (int): Number of vertices in the polygon
    """
    # Sample location
    if data == "brain":
        center = sample_location_brain_3d(volume)
    else:
        center = sample_location_abdom_3d(volume)

    # Sample size
    d = volume.shape[-1]
    anomaly_size = np.random.uniform(size_range[0] * d, size_range[1] * d, 3).tolist()

    # center = [128, 128, 128]
    # print(center, anomaly_size)

    # Sample points in 3D
    points, normals = sample_polygon_points_3d(n_vertices, normalize=True)

    # Create a mesh from the sampled points and normals
    mesh = create_mesh(points, normals)

    # Scale and shift mesh
    transform_trimesh(mesh, scale=anomaly_size, shift=center)

    # Convert mesh to voxel grid
    voxel_grid = mesh.voxelized(pitch=1)

    # Turn voxel grid into a 3D volume
    mask = volume_from_voxels(voxel_grid, size=volume.shape)

    # Voxel_grid was only surface, now fill holes
    mask = ndimage.binary_fill_holes(mask)

    return mask


def patch_exchange_3d(vol1: np.ndarray, vol2: np.ndarray, mask: np.ndarray):
    """Create a sample where one patch is switched from another sample
    with a random interpolation factor

    Args:
        vol1 (np.ndarray): shape [slices, w, h]
        vol2 (np.ndarray): shape [slices, w, h]
        mask (np.ndarray): shape [slices, w, h], mask indicating the voxels to swap
    """
    zero_mask = (1 - mask).astype(np.bool)

    # Sample interpolation factor alpha
    alpha = random.uniform(0.05, 0.95)
    # alpha = 0.95  # TODO: Remove

    # Target pixel value is also alpha
    patch = (mask * alpha).astype(np.float32)
    patch_inv = mask - patch

    # Interpolate between patches
    patch_set = patch * vol2 + patch_inv * vol1
    patchex = vol1 * zero_mask + patch_set

    valid_label = (
        mask * vol1)[..., None] != (mask * vol2)[..., None]
    valid_label = np.any(valid_label, axis=-1)

    label = valid_label * patch

    return patchex, label


def random_crop_centers_3D(volume: torch.Tensor, data: str, n_crops: int):
    """Sample random 3D coordinates in a volume as crop centers.

    Args:
        volume (torch.Tensor): Input volume
        data (str): Either "brain" or "abdom"
        n_crops (int): Number of random crops

    Returns:
        centers (torch.Tensor): Center coordinates for crops. Shape [n_crops, 3]
    """
    # Select locations for cropping
    if data == "brain":
        obj_inds = torch.nonzero(volume)
        if len(obj_inds) == 0:
            centers = torch.stack([
                torch.randint(0, d, (n_crops,)) for d in volume.shape
            ], dim=1)
        else:
            location_inds = torch.randint(0, obj_inds.shape[0], (n_crops,))
            centers = obj_inds[location_inds]
    elif data == "abdom":
        core_percent = 0.8
        dims = torch.tensor(volume.shape)
        core = (dims * core_percent).int().tolist()
        offset = ((1 - core_percent) * dims / 2).int().tolist()
        centers = torch.stack([
            torch.randint(offset[0], offset[0] + core[0], (n_crops,)),
            torch.randint(offset[1], offset[1] + core[1], (n_crops,)),
            torch.randint(offset[2], offset[2] + core[2], (n_crops,)),
        ], dim=1)
    else:
        raise ValueError("Incorrect value for data. Select 'brain' or 'abdom'")

    return centers


def random_crop_3D(volume: torch.Tensor, size: int, centers: torch.Tensor):
    """Get a random crop from a volume. Consider only the same locations as
    centers where anomalies are sampled from.

    Either centers must be not None or data AND n_crops must be not None

    Args:
        volume (torch.Tensor): Input volume to crop from
        size (int): Size of crop
        centers (torch.Tensor): Optional, centers can be already passed

    Returns:
        crops (torch.Tensor): Random 3D crops from volume at centers with size
    """
    # Lower and upper indices of patches
    neg = size // 2
    pos = size // 2
    if size % 2 != 0:
        pos += 1
    lower = centers - neg
    upper = centers + pos

    crops = []
    for lo, up in zip(lower, upper):
        # Get padding parameters
        front, top, left = torch.where(lo < 0, -lo, 0)
        back, bottom, right = (up - torch.tensor(volume.shape)).clip(0)
        pad = (left, right, top, bottom, front, back)

        # Clip lo to 0, because indexing fails with < 0
        lo = lo.clip(0)

        # Get crop
        inds = [slice(l, u) for l, u in zip(lo, up)]
        crop = volume[inds]

        # Pad if necessary
        crop = F.pad(crop, pad)
        crops.append(crop)

    return torch.stack(crops, dim=0)


if __name__ == "__main__":
    path = "/home/felix/datasets/MOOD/brain/train/00001.nii.gz"
    print(f"Loading {path}")
    volume1 = process_scan(path)
    path = "/home/felix/datasets/MOOD/brain/train/00011.nii.gz"
    print(f"Loading {path}")
    volume2 = process_scan(path)

    print("Generating patch mask")
    for _ in range(1000):
        ts = time()
        mask = sample_polygon_3d(volume1, size_range=[.1, .5], data="brain", n_vertices=30)
        print(f"Generated patch mask in {time() - ts:.4f}s")

    patchex, label = patch_exchange_3d(volume1, volume2, mask)
    volume_viewer(label)
    volume_viewer(patchex)

    import IPython ; IPython.embed() ; exit(1)
