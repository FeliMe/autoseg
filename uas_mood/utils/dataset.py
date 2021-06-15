from abc import abstractclassmethod
import argparse
from glob import glob
from multiprocessing import Pool
import os

import numpy as np
from torch.utils.data import Dataset

from uas_mood.utils.data_utils import load_segmentation, process_scan


DATAROOT = os.environ.get("DATAROOT")
assert DATAROOT is not None
MOODROOT = os.path.join(DATAROOT, "MOOD")


class PreloadDataset(Dataset):
    def __init__(self):
        super().__init__()

    @staticmethod
    @abstractclassmethod
    def load_batch():
        pass

    def load_to_ram(self, paths, img_size, slices_lower_upper):
        # Set number of cpus used
        num_cpus = os.cpu_count() - 4

        # Split list into batches
        batches = [list(p) for p in np.array_split(
            paths, num_cpus) if len(p) > 0]

        # Start multiprocessing
        with Pool(processes=num_cpus) as pool:
            res = pool.starmap(
                self.load_batch,
                zip(batches, [img_size for _ in batches], [slices_lower_upper for _ in batches])
            )

        return res


class TrainDataset(PreloadDataset):
    def __init__(self, files, img_size, slices_lower_upper):
        super().__init__()
        res = self.load_to_ram(files, img_size, slices_lower_upper)
        samples = [s for r in res for s in r]
        self.samples = [sl for sample in samples for sl in sample]

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_batch(files, img_size, slices_lower_upper):
        samples = []
        for f in files:
            # Samples are shape [width, height, slices]
            samples.append(process_scan(f, img_size, equalize_hist=True,
                                        slices_lower_upper=slices_lower_upper))

        return samples

    def __getitem__(self, idx):
        # Select sample
        sample = self.samples[idx]
        # Add fake channels dimension
        sample = sample.unsqueeze(0)
        return sample


class TestDataset(PreloadDataset):
    def __init__(self, files, img_size, slices_lower_upper):
        super().__init__()
        res = self.load_to_ram(files, img_size, slices_lower_upper)
        self.samples = [s for t in res for s in t["samples"]]
        self.segmentations = [s for t in res for s in t["segmentations"]]

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_batch(files, img_size, slices_lower_upper):
        samples = []
        segmentations = []
        for f in files:
            # Samples are shape [width, height, slices]
            samples.append((f, process_scan(f, img_size, equalize_hist=True,
                                        slices_lower_upper=slices_lower_upper)))
            # Load segmentation
            f_seg = f.split(".nii.gz")[0] + "_segmentation.nii.gz"
            segmentations.append(
                (f_seg, load_segmentation(f_seg, img_size,
                                  slices_lower_upper=slices_lower_upper)))

        return {
            "samples": samples,
            "segmentations": segmentations
        }

    def __getitem__(self, idx):
        return self.samples[idx], self.segmentations[idx]

    
class PatchSwapDataset(PreloadDataset):
    def __init__(self, files, img_size, slices_lower_upper):
        super().__init__()
        res = self.load_to_ram(files, img_size, slices_lower_upper)
        samples = [s for r in res for s in r]
        self.samples = [sl for sample in samples for sl in sample]

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_batch(files, img_size, slices_lower_upper):
        samples = []
        for f in files:
            # Samples are shape [width, height, slices]
            samples.append(process_scan(f, img_size, equalize_hist=True,
                                        slices_lower_upper=slices_lower_upper))

        return samples

    def __getitem__(self, idx):
        # Select sample
        sample = self.samples[idx]
        # Randomly select another sample
        # Randomly select a patch inside the samples anatomy
        # Sample interpolation factor alpha
        # Swap sample at patch
        # Patch location should be 0 in the background
        # Add fake channels dimension
        sample = sample.unsqueeze(0)
        return sample


def get_train_files(root : str, body_region : str):
    """Return all training files

    Args:
        root (str): Smth like $DATAROOT/MOOD/
        body_region (str): One of "brain" or "abdom"
    """
    assert body_region in ["brain", "abdom"]
    return glob(f"{os.path.join(root, body_region, 'train')}/?????.nii.gz")


def get_test_files(root : str, body_region : str):
    """Return all validation or test files

    Args:
        root (str): Smth like $DATAROOT/MOOD/brain/
        body_region (str): One of "brain" or "abdom"
        mode (str): One of "val" or "test"
    """
    assert body_region in ["brain", "abdom"]

    all_files = glob(f"{os.path.join(root, body_region, 'test')}/?????_*.nii.gz")
    files = []
    for f in all_files:
        if not f.endswith("_segmentation.nii.gz"):
            files.append(f)

    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    train_files = get_train_files(MOODROOT, "brain")
    val_files = get_test_files(MOODROOT, "brain", "val")
    print(f"# train_files: {len(train_files)}")
    print(f"# val_files: {len(val_files)}")

    # ----- TrainDataset -----
    # ds = TrainDataset(train_files[:10], 128, slices_lower_upper=[127, 131])
    # x = next(iter(ds))
    # print(x.shape)

    # ----- TestDataset -----
    # ds = TestDataset(val_files[:10], 128, slices_lower_upper=[127, 131])
    # x, y = next(iter(ds))
    # print(x[1].shape, y[1].shape)

    # ----- PatchSwapDataset -----
