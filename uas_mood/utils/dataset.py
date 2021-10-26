from abc import abstractclassmethod
from glob import glob
from multiprocessing import Pool
import os
import random
from typing import List
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

from uas_mood.utils.artificial_anomalies import patch_exchange, sample_complete_mask
from uas_mood.utils.data_utils import (
    load_image,
    load_segmentation,
    process_scan,
    volume_viewer,
)
from uas_mood.utils.utils import read_list_file


DATAROOT = os.environ.get("DATAROOT")
assert DATAROOT is not None
MOODROOT = os.path.join(DATAROOT, "MOOD")
CXR14ROOT = os.path.join(DATAROOT, "CXR8")


def plot(images, f=None):
    if not isinstance(images, list):
        images = [images]
    n = len(images)
    fig = plt.figure(figsize=(4 * n, 4))
    plt.axis("off")
    for i, im in enumerate(images):
        fig.add_subplot(1, n, i + 1)
        plt.imshow(im, cmap="gray", vmin=0., vmax=1.)

    if f is None:
        plt.show()
    else:
        plt.savefig(f)


class PreloadDataset(Dataset):
    def __init__(self):
        super().__init__()

    @staticmethod
    @abstractclassmethod
    def load_batch():
        pass

    def load_files_to_ram(self, paths, img_size):
        # Set number of cpus used
        num_cpus = os.cpu_count() - 4

        # Split list into batches
        batches = [list(p) for p in np.array_split(
            paths, num_cpus) if len(p) > 0]

        # Start multiprocessing
        with Pool(processes=num_cpus) as pool:
            res = pool.starmap(
                self.load_batch,
                zip(batches, [img_size for _ in batches])
            )

        return res


class TestDataset(PreloadDataset):
    def __init__(self, files, img_size):
        super().__init__()
        res = self.load_files_to_ram(files, img_size)
        samples = [s for t in res for s in t["samples"]]
        segmentations = [s for t in res for s in t["segmentations"]]

        self.samples = [(s[0], torch.from_numpy(s[1])) for s in samples]
        self.segmentations = [(s[0], torch.from_numpy(s[1]))
                              for s in segmentations]

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_batch(files, img_size):
        samples = []
        segmentations = []

        for f in files:
            # Samples are shape [width, height, slices]
            samples.append((f, process_scan(f, img_size, equalize_hist=False)))
            # Load segmentation, is in folder test_label/pixel instead of test
            f_seg = f.replace("test", "test_label/pixel")
            segmentations.append(
                (f_seg, load_segmentation(f_seg, img_size)))

        return {
            "samples": samples,
            "segmentations": segmentations
        }

    def __getitem__(self, idx):
        return self.samples[idx], self.segmentations[idx]


class PatchSwapDataset(PreloadDataset):
    def __init__(self, files, img_size, data, slices_on_forward, num_anomalies=1):
        super().__init__()
        assert data in ["brain", "abdom"]
        assert slices_on_forward in [1, 3], "PatchSwapDataset only works with slices_on_forward 1 or 3"

        self.num_anomalies = num_anomalies

        res = self.load_files_to_ram(files, img_size)
        samples = [s for r in res for s in r]
        # Samples: list of patient volumes [slices, w, h]

        # Number of scans in dataset
        self.n_scans = len(samples)
        # Number of slices per scan (3 viewing directions)
        # self.n_slices = np.array(samples[0].shape).sum()
        self.n_slices = -1
        # Number of slices in one viewing direction
        self.sample_depth = samples[0].shape[0]

        self.slices_on_forward = slices_on_forward
        self.mid_slice = slices_on_forward // 2

        self.samples = []
        for sample in samples:
            # Add slices from all three viewing directions
            axial = sample
            coronal = np.moveaxis(sample, 1, 0)
            sagittal = np.moveaxis(sample, 2, 0)
            self.samples += [sl for sl in axial]
            self.samples += [sl for sl in coronal]
            self.samples += [sl for sl in sagittal]
            if self.n_slices == -1:
                self.n_slices = len(self.samples)
        self.data = data

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_batch(files, img_size):
        samples = []
        for f in files:
            # Samples are shape [width, height, slices]
            samples.append(process_scan(f, img_size, equalize_hist=False))
            if np.any(np.isnan(samples[-1])):
                print(f)

        return samples

    def create_anomaly(self, img1, img2):
        """Create a sample where one patch is switched from another sample
        with a random interpolation factor

        Args:
            img1 (torch.Tensor): shape [w, h]
            img2 (torch.Tensor): shape [w, h]
        """
        size_range = [.1, .5] if self.data == "brain" else [.2, .6]

        # Sample an anomaly mask
        mask = sample_complete_mask(
            n_patches=self.num_anomalies, blur_prob=0., img=img1,
            size_range=size_range, data=self.data, patch_type="polygon",
            poly_type="cubic", n_vertices=10
        )

        # Swap patches between img1 and img2 at mask
        patchex, label = patch_exchange(img1, img2, mask)
        # Select mask of middle slice as target
        label = label[self.mid_slice][None]

        # Convert to tensor
        patchex = torch.from_numpy(patchex)
        label = torch.from_numpy(label)

        return patchex, label

    def __getitem__(self, idx):
        if self.slices_on_forward == 3:
            # If idx is a border slice, select next of previous one
            if idx % self.sample_depth == 0:
                idx += 1  # Lower border, select next idx
            if (idx % self.sample_depth) % (self.sample_depth - 1) == 0:
                idx -= 1  # Upper border, select prev idx

        # Select sample
        lo = self.slices_on_forward // 2
        hi = self.slices_on_forward // 2 + 1
        sample = np.stack(self.samples[idx - lo:idx + hi]).copy()

        # Randomly select another sample at the same slice
        i_slice = idx % self.n_slices
        other_scan = random.randint(0, self.n_scans - 1)
        other_idx = other_scan * self.n_slices + i_slice
        other_sample = np.stack(self.samples[other_idx - lo:other_idx + hi])

        # Create foreign patch interpolation
        sample, patch = self.create_anomaly(sample, other_sample)

        return sample, patch


class CXR14PatchSwapDataset(Dataset):
    def __init__(self, files: List[str], img_size: int, load_to_ram: bool = True,
                 anomaly_shape: str = "polygon", num_anomalies: int = 1):
        super().__init__()
        self.samples = files
        self.img_size = img_size
        self.load_to_ram = load_to_ram
        self.anomaly_shape = anomaly_shape
        self.num_anomalies = num_anomalies
        print(f"Using {anomaly_shape}s as anomalies")

        if self.load_to_ram:
            warn("The functionality of this class only works with num_workers=0 in the DataLoader")

        self.left_to_load = len(self.samples)

    def load_sample(self, sample):
        if isinstance(sample, str):
            sample = load_image(sample, self.img_size).numpy()
            self.left_to_load -= 1
            return sample
        else:
            return sample


    def __len__(self):
        return len(self.samples)

    def create_anomaly(self, img1, img2):
        """Create a sample where one patch is switched from another sample
        with a random interpolation factor

        Args:
            img1 (torch.Tensor): shape [w, h]
            img2 (torch.Tensor): shape [w, h]
        """
        size_range = [.05, .7]

        # Sample an anomaly mask
        mask = sample_complete_mask(
            n_patches=self.num_anomalies, blur_prob=0., img=img1,
            size_range=size_range, data="cxr", patch_type=self.anomaly_shape,
            poly_type="cubic", n_vertices=10
        )

        # Swap patches between img1 and img2 at mask
        patchex, label = patch_exchange(img1, img2, mask)

        # Convert to tensor
        patchex = torch.from_numpy(patchex)
        label = torch.from_numpy(label)

        return patchex, label

    def __getitem__(self, idx):
        # Load a sample
        self.samples[idx] = self.load_sample(self.samples[idx])
        sample = self.samples[idx]

        # Randomly select another sample at the same slice
        other_idx = random.randint(0, len(self) - 1)
        self.samples[other_idx] = self.load_sample(self.samples[other_idx])
        other_sample = self.samples[other_idx]

        # Create foreign patch interpolation
        sample, patch = self.create_anomaly(sample, other_sample)

        return sample, patch


class CXR14TestDataset(Dataset):
    def __init__(self, files, labels, img_size):
        super().__init__()
        self.samples = files
        self.labels = labels
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename = self.samples[idx]
        sample = load_image(self.samples[idx], self.img_size)
        label = self.labels[idx]
        return sample, filename, label


def get_train_files(root: str, body_region: str):
    """Return all training files

    Args:
        root (str): Smth like $DATAROOT/MOOD/
        body_region (str): One of "brain" or "abdom"
    """
    assert body_region in ["brain", "abdom"]
    return glob(f"{os.path.join(root, body_region, 'train')}/?????.nii.gz")


def get_test_files(root: str, body_region: str):
    """Return all validation or test files

    Args:
        root (str): Smth like $DATAROOT/MOOD/brain/
        body_region (str): One of "brain" or "abdom"
        mode (str): One of "val" or "test"
    """
    assert body_region in ["brain", "abdom"]
    return glob(f"{os.path.join(root, body_region, 'test')}/?????_*.nii.gz")


if __name__ == '__main__':
    data = "brain"
    # data = "abdom"
    img_size = 512 if data == "abdom" else 256
    train_files = get_train_files(MOODROOT, data)
    test_files = get_test_files(MOODROOT, data)
    print(f"# train_files: {len(train_files)}")
    print(f"# test_files: {len(test_files)}")

    # ----- TestDataset -----
    # ds = TestDataset(test_files[:10], 256)
    # idx = 103
    # x, y = next(iter(ds))
    # print(x[1].shape, y[1].shape)
    # plot([x[1][idx], y[1][idx]])

    # ----- PatchSwapDataset -----
    # ds = PatchSwapDataset(train_files[:10], img_size,
    #                       data=data, slices_on_forward=3)
    # x, y = ds.__getitem__(128)
    # print(x.shape, y.shape, y.max())
    # print(x.dtype, y.dtype)
    # x = x[1].unsqueeze(0)
    # plot([x[0], (x[0] + y[0]).clip(0, 1), y[0]], f="fig.png")

    # ----- CXR14PatchSwapDataset -----
    img_size = 256
    train_files = read_list_file(os.path.join(CXR14ROOT, "train_lists/norm_train_list.txt"))
    train_files = [os.path.join(CXR14ROOT, "images", f) for f in train_files]
    train_files = train_files[:1000]
    ds = CXR14PatchSwapDataset(train_files, img_size=img_size)
    x, y = next(iter(ds))
    plot([x[0], y[0]])
    import IPython ; IPython.embed() ; exit(1)
