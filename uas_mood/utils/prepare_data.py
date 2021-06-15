from collections import defaultdict
from glob import glob
import os
import random
import shutil

import numpy as np
from tqdm import tqdm

from uas_mood.utils.artificial_anomalies import create_random_anomaly
from uas_mood.utils.data_utils import load_nii, save_nii

""" Global variables """

TRAINFRAC = 0.6  # 60 % of the total files are used for training, 40% for testing
TESTNORMALFRAC = 0.25  # 25% of test samples don't get artificial anomalies

DATAROOT = os.environ.get('DATAROOT')
assert DATAROOT is not None, "Set the environment variable DATAROOT to your datasets folder"

MOODROOT = os.path.join(DATAROOT, "MOOD")
ABDOMROOT = os.path.join(MOODROOT, "abdom")
BRAINROOT = os.path.join(MOODROOT, "brain")
ABDOMTRAIN = os.path.join(ABDOMROOT, "train")
BRAINTRAIN = os.path.join(BRAINROOT, "train")

FOLDERSTRUC = """Initial folder structure of the MOOD folder:
Data available at: https://www.synapse.org/#!Synapse:syn21343101/wiki/599515

MOOD
|-- abdom
|   |-- train
|   |   |-- 00000.nii.gz
|   |   |-- 00001.nii.gz
|   |   `-- ...
|   |-- toy
|   |   |-- toy_0.nii.gz
|   |   |-- toy_1.nii.gz
|   |   `-- ...
|   `-- toy_label
|       |-- pixel
|       |   |-- toy_0.nii.gz
|       |   |-- toy_1.nii.gz
|       |   `-- ...
|       `-- sample
|           |-- toy_0.nii.gz.txt
|           |-- toy_1.nii.gz.txt
|           `-- ...
´-- brain
    |-- train
    |   |-- 00000.nii.gz
    |   |-- 00001.nii.gz
    |   `-- ...
    |-- toy
    |   |-- toy_0.nii.gz
    |   |-- toy_1.nii.gz
    |   `-- ...
    `-- toy_label
        |-- pixel
        |   |-- toy_0.nii.gz
        |   |-- toy_1.nii.gz
        |   `-- ...
        `-- sample
            |-- toy_0.nii.gz.txt
            |-- toy_1.nii.gz.txt
            `-- ...
"""


def sanity_check():
    # Rename train folders if necessary
    abdom_train_ = os.path.join(ABDOMROOT, "abdom_train")
    brain_train_ = os.path.join(BRAINROOT, "brain_train")
    if os.path.isdir(abdom_train_):
        print(f"Renaming {abdom_train_} to {ABDOMTRAIN}")
        os.rename(abdom_train_, ABDOMTRAIN)
    if os.path.isdir(brain_train_):
        print(f"Renaming {brain_train_} to {BRAINTRAIN}")
        os.rename(brain_train_, BRAINTRAIN)

    # Check folder structure
    folder_ok = True
    folder_ok &= os.path.isdir(ABDOMTRAIN)
    folder_ok &= os.path.isdir(BRAINTRAIN)
    assert folder_ok, FOLDERSTRUC

    # Check if all files are present
    n_abdom_train = len(glob(f"{ABDOMTRAIN}/?????.nii.gz"))
    assert n_abdom_train == 550, f"Missing files in {ABDOMTRAIN}, only found {n_abdom_train}"
    n_brain_train = len(glob(f"{BRAINTRAIN}/?????.nii.gz"))
    assert n_brain_train == 800, f"Missing files in {BRAINTRAIN}, only found {n_brain_train}"

    print("Sanity check successfull!")


def split_ds(root):
    # Init names
    train_folder = os.path.join(root, "train")
    test_folder = os.path.join(root, "test")

    # Create val- and test-folder
    os.makedirs(test_folder, exist_ok=True)

    # Get all files
    files = sorted(glob(f"{train_folder}/?????.nii.gz"))
    n_files = len(files)

    # Split files
    train_idx = int(n_files * TRAINFRAC)
    test_files = files[train_idx:]

    # Move files to respective directories
    for src in test_files:
        dst = os.path.join(test_folder, src.split('/')[-1])
        shutil.move(src, dst)
    print(f"Moved {len(test_files)} fo {test_folder}")


def create_test_anomalies(root):
    counts = defaultdict(int)
    files = sorted(glob(f"{root}/?????.nii.gz"))
    n_files = len(files)

    random.shuffle(files)
    normal_idx = round(n_files * TESTNORMALFRAC)
    normal_files = files[:normal_idx]
    anomal_files = files[normal_idx:]

    # Create files without anomalies
    print("Creating files without anomalies")
    for f in tqdm(normal_files):
        # Load volume
        volume, affine = load_nii(f, dtype="float64")
        # Segmentation is all 0
        segmentation = np.zeros_like(volume)
        # Save
        target = f"{f.split('.nii.gz')[0]}_normal.nii.gz"
        seg_target = f"{f.split('.nii.gz')[0]}_normal_segmentation.nii.gz"
        save_nii(target, volume, affine, dtype= "float32")
        save_nii(seg_target, segmentation, affine, dtype= "short")

    # Create files with anomalies
    print("Creating files with anomalies")
    pbar = tqdm(anomal_files)
    for f in pbar:
        # Load volume
        volume, affine = load_nii(f, dtype="float64")
        # Create anomaly
        anomaly, segmentation, anomaly_type, _, _ = create_random_anomaly(volume)
        # Update statistics
        pbar.set_description(anomaly_type)
        counts[anomaly_type] += 1
        # Save
        target = f"{f.split('.nii.gz')[0]}_{anomaly_type}.nii.gz"
        seg_target = f"{f.split('.nii.gz')[0]}_{anomaly_type}_segmentation.nii.gz"
        save_nii(target, anomaly, affine, dtype= "float32")
        save_nii(seg_target, segmentation, affine, dtype= "short")

    for k, v in counts.items():
        print(f"{k}: absolute {v}, relative {v / n_files:.3f}")


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    # Check if data is correctly downloaded
    # sanity_check()
    # print("Splitting abdom files")
    # split_ds(ABDOMROOT)
    # print("Splitting brain files")
    # split_ds(BRAINROOT)
    # print("Creating artificial anomalies for brain test")
    # create_test_anomalies(os.path.join(BRAINROOT, "test"))
    print("Creating artificial anomalies for abdomen test")
    create_test_anomalies(os.path.join(ABDOMROOT, "test"))
