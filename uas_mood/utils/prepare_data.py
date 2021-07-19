import argparse
from collections import defaultdict
from glob import glob
import os
import random
import shutil

import numpy as np
from tqdm import tqdm

from uas_mood.utils.test_anomalies import create_random_anomaly
from uas_mood.utils.data_utils import load_nii, save_nii
from uas_mood.utils.utils import write_file

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
    test_folder = os.path.join(root, "test_raw")

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


def create_test_anomalies(root_dir, target_dir, segmentation_dir, label_dir):
    counts = defaultdict(int)
    files = sorted(glob(f"{root_dir}/?????.nii.gz"))
    n_files = len(files)

    random.shuffle(files)
    normal_idx = round(n_files * TESTNORMALFRAC)
    normal_files = files[:normal_idx]
    anomal_files = files[normal_idx:]

    # Create target_dir and segmentation_dir
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(segmentation_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # Create files without anomalies
    print("Creating files without anomalies")
    for f in tqdm(normal_files):
        # Load volume
        volume, affine = load_nii(f, dtype="float64")
        # Segmentation is all 0
        segmentation = np.zeros_like(volume)
        # Save
        name = f.split('/')[-1].split('.')[0]
        target = os.path.join(target_dir, f"{name}_normal.nii.gz")
        seg_target = os.path.join(segmentation_dir, f"{name}_normal.nii.gz")
        label_target = os.path.join(label_dir, f"{name}_normal.nii.gz.txt")
        save_nii(target, volume, affine, dtype="float32")
        save_nii(seg_target, segmentation, affine, dtype="short")
        write_file(label_target, str(0))

    # Create files with anomalies
    print("Creating files with anomalies")
    pbar = tqdm(anomal_files)
    for f in pbar:
        # Load volume
        volume, affine = load_nii(f, dtype="float64")
        # Create anomaly
        anomaly, segmentation, anomaly_type, _, _ = create_random_anomaly(
            volume)
        # Update statistics
        pbar.set_description(anomaly_type)
        counts[anomaly_type] += 1
        # Save
        name = f.split('/')[-1].split('.')[0]
        target = os.path.join(target_dir, f"{name}_{anomaly_type}.nii.gz")
        seg_target = os.path.join(
            segmentation_dir, f"{name}_{anomaly_type}.nii.gz")
        label_target = os.path.join(
            label_dir, f"{name}_{anomaly_type}.nii.gz.txt")
        save_nii(target, anomaly, affine, dtype="float32")
        save_nii(seg_target, segmentation, affine, dtype="short")
        write_file(label_target, str(1))

    for k, v in counts.items():
        print(f"{k}: absolute {v}, relative {v / n_files:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--create_anomalies", action="store_true")
    parser.add_argument("--data", type=str,
                        choices=["brain", "abdom"], required=True)
    args = parser.parse_args()

    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    if args.split:
        # Check if data is correctly downloaded
        sanity_check()
        if args.data == "abdom":
            print("Splitting abdom files")
            split_ds(ABDOMROOT)
        else:
            print("Splitting brain files")
            split_ds(BRAINROOT)

    if args.create_anomalies:
        if args.data == "abdom":
            print("Creating artificial anomalies for abdomen test")
            create_test_anomalies(os.path.join(ABDOMROOT, "test_raw"),
                                  os.path.join(ABDOMROOT, "test"),
                                  os.path.join(
                                      ABDOMROOT, "test_label", "pixel"),
                                  os.path.join(ABDOMROOT, "test_label", "sample"))
        else:
            print("Creating artificial anomalies for brain test")
            create_test_anomalies(os.path.join(BRAINROOT, "test_raw"),
                                  os.path.join(BRAINROOT, "test"),
                                  os.path.join(
                                      BRAINROOT, "test_label", "pixel"),
                                  os.path.join(BRAINROOT, "test_label", "sample"))
