import os
import shutil

from glob import glob

""" Global variables """

TRAINFRAC = 0.6
VALFRAC = 0.3
TESTFRAC = 0.1

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
    val_folder = os.path.join(root, "val")
    test_folder = os.path.join(root, "test")

    # Create val- and test-folder
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Get all files
    files = sorted(glob(f"{train_folder}/?????.nii.gz"))
    n_files = len(files)

    # Split files
    train_idx = int(n_files * TRAINFRAC)
    val_idx = train_idx + int(n_files * VALFRAC)
    val_files = files[train_idx:val_idx]
    test_files = files[val_idx:]

    # Move files to respective directories
    for src in val_files:
        dst = os.path.join(val_folder, src.split('/')[-1])
        shutil.move(src, dst)
    print(f"Moved {len(val_files)} fo {val_folder}")

    for src in test_files:
        dst = os.path.join(test_folder, src.split('/')[-1])
        shutil.move(src, dst)
    print(f"Moved {len(test_files)} fo {test_folder}")


if __name__ == '__main__':
    # Check if data is correctly downloaded
    sanity_check()
    print("Splitting abdom files")
    split_ds(ABDOMROOT)
    print("Splitting brain files")
    split_ds(BRAINROOT)
