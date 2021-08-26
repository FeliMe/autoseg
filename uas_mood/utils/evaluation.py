import argparse
from glob import glob
from multiprocessing import Pool
import os
from time import time

import numpy as np
from skimage import measure
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
import torch
from torchvision.utils import make_grid
from tqdm import tqdm
from tqdm import tqdm

from uas_mood.utils import data_utils, utils

# matplotlib can't be imported in a read-only filesystem
try:
    import matplotlib.pyplot as plt
except FileNotFoundError:
    pass

def plot_results(images: list, titles: list, n_images=20):
    """Returns a plot containing the input images, reconstructed images,
    uncertainty maps and anomaly maps"""

    if len(images) != len(titles):
        raise RuntimeError("not the same number of images and titles")

    import IPython ; IPython.embed()
    # Stack tensors to image grid and transform to numpy for plotting
    img_dict = {}
    for img, title in zip(images, titles):
        assert img[0].ndim == 3, "Invalid number of dimensions, missing channel dim?"
        img_grid = make_grid(img[:n_images].float(), nrow=1, normalize=False)
        img_grid = utils.torch2np_img(img_grid)
        img_dict[title] = img_grid

    n = len(images)

    # Construct matplotlib figure
    fig = plt.figure(figsize=(3 * n, n_images))
    plt.axis('off')
    for i, key in enumerate(img_dict.keys()):
        a = fig.add_subplot(1, n, i + 1)
        plt.imshow(img_dict[key], vmin=0., vmax=1.)
        a.set_title(key)

    return fig


def compute_average_precision(predictions, targets):
    """Compute Average Precision

    Args:
        predictions (torch.Tensor): Anomaly scores
        targets (torch.Tensor): Segmentation map, must be binary
    """
    if (targets - targets.int()).sum() > 0.:
        raise RuntimeError("targets for AP must be binary")
    ap = average_precision_score(targets.reshape(-1), predictions.reshape(-1))
    return ap


def samplewise_score(pred) -> float:
    """Compute the anomaly score for a patient volume from the predictions of
    the network.

    Args:
        pred (np.ndarray or torch.Tensor): Network output for one patient volume
                                           of shape [slices, w, h]
    Returns:
        samplewise_score (float): Anomaly score for that patient
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()

    im_level_score = np.mean(pred, axis=(1, 2))
    im_level_score_s = sorted(im_level_score)

    # Take mean of highest 90% of values
    im_level_score_s = im_level_score_s[int(len(im_level_score_s) * 0.9):]
    samplewise_score = np.mean(im_level_score_s)

    return samplewise_score


def samplewise_score_list(predictions):
    """Return sample-wise anomaly scores for a list or tensor of patient volumes
    each with shape [slices, w, h]

    Args:
        predictions (list, or iterable of tensors or array)
    """
    return torch.tensor([samplewise_score(pred) for pred in predictions])


def evaluate_sample_wise(predictions, targets, verbose=True):
    auroc = roc_auc_score(targets, predictions)
    ap = average_precision_score(targets, predictions)

    if verbose:
        print(f"AUROC: {auroc:.4f}")
        print(f"AP: {ap:.4f}")

    return auroc, ap


def evaluate_pixel_wise(predictions, targets, verbose=True):
    # ap = compute_average_precision(predictions, targets)
    ap = np.mean([compute_average_precision(p, t) for p, t in zip(predictions, targets)])

    if verbose:
        print(f"AP: {ap:.4f}")

    return ap


def full_evaluation_sample(predictions, targets, anomalies):
    """Perform the full sample-wise evaluation for every anomaly type

    Args:
        predictions (torch.Tensor): Predicted labels, shape [n]
        targets (torch.Tensor): Target labels, shape [n]
        anomalies (list of str): Anomaly types, len = n
    """
    unique_anomalies = set(anomalies)
    unique_anomalies.discard("normal")
    unique_anomalies.discard("")

    for anomaly in sorted(unique_anomalies):
        # Filter only relevant anomalies (and "normal")
        considered = [anomaly, "normal"]
        p = torch.tensor(
            [m for m, a in zip(predictions, anomalies) if a in considered])
        t = torch.tensor(
            [l for l, a in zip(targets, anomalies) if a in considered])

        # Evaluate sample-wise
        print(f"\nEvaluating performance on {anomaly} with {len(p)} samples")
        evaluate_sample_wise(p, t, verbose=True)

    print(f"\nEvaluating total performance with {len(predictions)} samples")
    evaluate_sample_wise(predictions, targets, verbose=True)


def full_evaluation_pixel(predictions, targets, anomalies):
    """Perform the full sample-wise evaluation for every anomaly type

    Args:
        predictions (torch.Tensor): Predicted anomaly maps, shape [n, slices, w, h]
        targets (torch.Tensor): Target segmentations, shape [n, slices, w, h]
        anomalies (list of str): Anomaly types, len = n
    """
    unique_anomalies = set(anomalies)
    unique_anomalies -= {"normal", ""}
    # unique_anomalies.discard("normal")
    # unique_anomalies.discard("")

    if not len(unique_anomalies) == 1:
        for anomaly in sorted(unique_anomalies):
            print(f"\nEvaluating performance on {anomaly}")
            t_start = time()

            # Filter only relevant anomalies
            p = torch.stack(
                [m for m, a in zip(predictions, anomalies) if a == anomaly])
            t = torch.stack([l for l, a in zip(targets, anomalies) if a == anomaly])

            # Evaluate sample-wise
            evaluate_pixel_wise(p, t)
            print(f"Time: {time() - t_start:.2f}s")

    print("\nEvaluating total performance")
    t_start = time()
    if not len(unique_anomalies) == 1:
        p = torch.stack([m for m, a in zip(predictions, anomalies) if a in unique_anomalies])
        t = torch.stack([l for l, a in zip(targets, anomalies) if a in unique_anomalies])
        evaluate_pixel_wise(p, t)
    else:
        print(f"{compute_average_precision(predictions, targets):.4f}")
    print(f"Time: {time() - t_start:.2f}s")


def full_evaluation_pixel_memory_efficient(pred_files, target_files, anomalies, n_proc=1):
    def ap_from_files(files):
        pred_file, target_file = files
        # Load files
        pred = data_utils.load_nii(pred_file)[0]
        target = data_utils.load_nii(target_file, size=pred.shape[-2], dtype="short")[0]
        # Evaluate
        return compute_average_precision(
            torch.from_numpy(pred),
            torch.from_numpy(target)
        )

    unique_anomalies = set(anomalies)
    unique_anomalies.discard("normal")
    unique_anomalies.discard("")

    for anomaly in sorted(unique_anomalies):
        t_start = time()
        ap = 0.

        # Filter only current anomaly type
        files = [(p, t) for p, t, a in zip(pred_files, target_files, anomalies) if a == anomaly]
        print(f"\nEvaluating performance on {anomaly} with {len(files)} samples")

        # Compute average precision on multiple cores
        pool = Pool(n_proc)
        ap = pool.map(ap_from_files, files)
        print(f"AP: {np.mean(ap):.4f}")
        print(f"Time: {time() - t_start:.2f}s")

    print("\nEvaluating total performance")
    t_start = time()
    ap = 0.

    # Filter out all normal samples, they would induce division by 0
    files = [(p, t) for p, t, a in zip(pred_files, target_files, anomalies) if a != "normal"]

    # Compute average precision on multiple cores
    pool = Pool(n_proc)
    # ap = pool.map(ap_from_files, files)
    ap = ap_from_files(files)
    print(f"AP: {np.mean(ap):.4f}")
    print(f"Time: {time() - t_start:.2f}s")


def eval_dir(pred_dir, target_dir, mode, n_proc=1):
    # List files from dir
    ext = "*.nii.gz" if mode == "pixel" else "*.txt"
    pred_files = glob(os.path.join(pred_dir, ext))
    # target_files = glob(os.path.join(target_dir, ext))
    target_files = [os.path.join(target_dir, p.split('/')[-1]) for p in pred_files]
    for f in target_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f'No such file {f} in target_dir')
    print(
        f"Found {len(pred_files)} predictions and {len(target_files)} targets.")
    assert len(pred_files) == len(target_files)

    # Get names of anomalies
    anomalies = [f.split('/')[-1].split(".nii.gz")[0][6:]
                 for f in target_files]
    print(f"Found anomalies: {set(anomalies)}")

    # Load from files
    if mode == "pixel":
        if len(set(anomalies)) == 1:
            print("Reading nii files with predictions")
            preds = np.stack([data_utils.load_nii(f)[0] for f in pred_files])
            pred_img_size = preds[0].shape[-2]
            print("Reading nii files with targets")
            targets = np.stack([data_utils.load_nii(f, size=pred_img_size, dtype="short")[0]
                                for f in target_files])
            full_evaluation_pixel(predictions=torch.from_numpy(preds),
                                targets=torch.from_numpy(targets),
                                anomalies=anomalies)
        else:
            full_evaluation_pixel_memory_efficient(pred_files, target_files,
                                                   anomalies, n_proc)
    elif mode == "sample":
        preds = [float(utils.read_file(f)) for f in pred_files]
        targets = [int(utils.read_file(f)) for f in target_files]
        full_evaluation_sample(predictions=preds,
                               targets=targets,
                               anomalies=anomalies)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate from directories")
    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="Directory with predicted anomaly maps or labels")
    parser.add_argument("-t", "--target_dir", type=str, required=True,
                        help="Directory with ground truth anomaly maps or labels")
    parser.add_argument("-m", "--mode", type=str, required=True,
                        help="Evaluation mode, choose between 'pixel' or 'sample'",
                        choices=["pixel", "sample"])
    parser.add_argument("--n_proc", type=int, default=20,
                        help="Number of processes")
    args = parser.parse_args()

    eval_dir(args.input_dir, args.target_dir, args.mode, args.n_proc)
