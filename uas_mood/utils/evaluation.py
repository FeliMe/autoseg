import argparse
from glob import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
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

from uas_mood.utils import data_utils, utils


def plot_results(images: list, titles: list, n_images=20):
    """Returns a plot containing the input images, reconstructed images,
    uncertainty maps and anomaly maps"""

    if len(images) != len(titles):
        raise RuntimeError("not the same number of images and titles")

    # Stack tensors to image grid and transform to numpy for plotting
    img_dict = {}
    for img, title in zip(images, titles):
        assert img[0].ndim == 3, "Invalid number of dimensions, missing channel dim?"
        img_grid = make_grid(img[:n_images].float(), nrow=1, normalize=False)
        # img_grid = make_grid(
        #     img[:n_images].float(), nrow=1, normalize=True, scale_each=True)
        img_grid = utils.torch2np_img(img_grid)
        img_dict[title] = img_grid

    n = len(images)

    # Construct matplotlib figure
    fig = plt.figure(figsize=(3 * n, 1.0 * n_images))
    plt.axis('off')
    for i, key in enumerate(img_dict.keys()):
        a = fig.add_subplot(1, n, i + 1)
        plt.imshow(img_dict[key], vmin=0., vmax=1.)
        a.set_title(key)

    return fig


def plot_prc(predictions, targets):
    """Returns a plot of the Precision-Recall Curve (PRC)

    Args:
        predictions (torch.Tensor): Predicted anomaly map
        targets (torch.Tensor): Target segmentation

    Returns:
        fig (matplotlib.figure.Figure): Finished plot
    """

    precision, recall, _ = precision_recall_curve(
        targets.view(-1), predictions.view(-1))
    ap = compute_average_precision(predictions, targets)
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    plt.title("Precision-Recall Curve")
    plt.plot(precision, recall, 'b', label='AP = %0.2f' % ap)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Recall')
    plt.xlabel('Precision')
    return fig


def compute_best_dice(preds, targets, n_thresh=100, verbose=True):
    """Compute the best dice score between an anomaly map and the ground truth
    segmentation using a greedy binary search with depth search_depth

    Args:
        preds (torch.tensor): Predicted binary anomaly map. Shape [b, c, h, w]
        targets (torch.tensor): Target label [b] or segmentation map [b, c, h, w]
        n_thresh (int): Number of thresholds to try
    Returns:
        max_dice (float): Maximal dice score
        max_thresh (float): Threshold corresponding to maximal dice score
    """
    thresholds = np.linspace(preds.min(), preds.max(), n_thresh)
    threshs = []
    scores = []
    for t in tqdm(thresholds, desc="DSC search", disable=not verbose):
        # for t in thresholds:
        dice = compute_dice(torch.where(preds > t, 1., 0.), targets)
        scores.append(dice)
        threshs.append(t)

    scores = torch.stack(scores, 0)
    max_thresh = threshs[scores.argmax()]

    # Get best dice once again after connected component analysis
    bin_preds = torch.where(preds > max_thresh, 1., 0.)
    bin_preds = utils.connected_components_3d(bin_preds)
    max_dice = compute_dice(bin_preds, targets)
    return max_dice, max_thresh


def compute_dice_fpr(preds, targets, max_fprs=[0.01, 0.05, 0.1]):
    fprs, _, thresholds = compute_roc(preds, targets)
    dices = []
    for max_fpr in max_fprs:
        th = thresholds[fprs < max_fpr][-1]
        bin_preds = torch.where(preds > th, 1., 0.)
        bin_preds = utils.connected_components_3d(bin_preds)
        dice = compute_dice(bin_preds, targets)
        dices.append(dice)
        print(f"DICE{int(max_fpr * 100)}: {dice:.4f}, threshold: {th:.4f}")
    return dices


def compute_dice(predictions, targets) -> float:
    """Compute the DICE score. This only works for segmentations.
    PREDICTIONS NEED TO BE BINARY!

    Args:
        predictions (torch.tensor): Predicted binary anomaly map. Shape [b, c, h, w]
        targets (torch.tensor): Target label [b] or segmentation map [b, c, h, w]
    Returns:
        dice (float)
    """
    if (predictions - predictions.int()).sum() > 0.:
        raise RuntimeError("predictions for DICE score must be binary")
    if (targets - targets.int()).sum() > 0.:
        raise RuntimeError("targets for DICE score must be binary")

    pred_sum = predictions.view(-1).sum()
    targ_sum = targets.view(-1).sum()
    intersection = predictions.view(-1).float() @ targets.view(-1).float()
    dice = (2 * intersection) / (pred_sum + targ_sum)
    return dice


def compute_pro_auc(predictions, targets, expect_fpr=0.3, max_steps=300):
    """Computes the PRO-score and intersection over union (IOU)
    Code from: https://github.com/YoungGod/DFR/blob/master/DFR-source/anoseg_dfr.py
    """

    def rescale(x):
        return (x - x.min()) / (x.max() - x.min())

    if torch.is_tensor(predictions):
        predictions = predictions.numpy()
    if torch.is_tensor(targets):
        targets = targets.numpy()

    # Squeeze away channel dimension
    predictions = predictions.squeeze(1)
    targets = targets.squeeze(1)

    # Binarize target segmentations
    targets[targets <= 0.5] = 0
    targets[targets > 0.5] = 1
    targets = targets.astype(np.bool)

    # Maximum and minimum thresholdsmax_th = scores.max()
    max_th = predictions.max()
    min_th = predictions.min()
    delta = (max_th - min_th) / max_steps

    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(predictions, dtype=np.bool)
    # for step in tqdm(range(max_steps), desc="PRO AUC"):
    for step in range(max_steps):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[predictions <= thred] = 0
        binary_score_maps[predictions > thred] = 1

        pro = []    # per region overlap
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        for i in range(len(binary_score_maps)):    # for i th image
            # pro (per region level)
            label_map = measure.label(targets[i], connectivity=2)
            props = measure.regionprops(label_map)
            for prop in props:
                # find the bounding box of an anomaly region
                x_min, y_min, x_max, y_max = prop.bbox
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                # cropped_mask = targets[i][x_min:x_max, y_min:y_max]   # bug!
                cropped_targets = prop.filled_image    # corrected!
                intersection = np.logical_and(
                    cropped_pred_label, cropped_targets).astype(np.float32).sum()
                pro.append(intersection / prop.area)

        # against steps and average metrics on the testing data
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr for pro-auc
        targets_neg = ~targets
        fpr = np.logical_and(
            targets_neg, binary_score_maps).sum() / targets_neg.sum()
        fprs.append(fpr)
        threds.append(thred)

    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)

    # default 30% fpr vs pro, pro_auc
    # find the indexs of fprs that is less than expect_fpr (default 0.3)
    idx = fprs <= expect_fpr
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)    # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]
    pro_auc_score = auc(fprs_selected, pros_mean_selected)
    print(f"PRO AUC ({int(expect_fpr*100)}% FPR): {pro_auc_score:.4f}")

    return pro_auc_score


def compute_average_precision(predictions, targets):
    """Compute Average Precision

    Args:
        predictions (torch.tensor): Anomaly scores
        targets (torch.tensor): Segmentation map, must be binary
    """
    if (targets - targets.int()).sum() > 0.:
        raise RuntimeError("targets for AP must be binary")
    ap = average_precision_score(targets.reshape(-1), predictions.reshape(-1))
    return ap


def fpi_sample_score(pred) -> float:
    """Compute the anomaly score for a patient volume from the predictions of
    the network.

    Args:
        pred (np.ndarray or torch.Tensor): Network output for one patient volume
                                           of shape [slices, w, h]
    Returns:
        sample_score (float): Anomaly score for that patient
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()

    im_level_score = np.mean(pred, axis=(1, 2))
    im_level_score_s = sorted(im_level_score)

    # Take mean of top 90% of values
    im_level_score_s = im_level_score_s[int(len(im_level_score_s) * 0.9):]
    sample_score = np.mean(im_level_score_s)

    return sample_score


def fpi_sample_score_list(predictions):
    """Return sample-wise anomaly scores for a list or tensor of patient volumes
    each with shape [slices, w, h]

    Args:
        predictions (list of tensors or array, or tensor or array)
    """
    return torch.tensor([fpi_sample_score(pred) for pred in predictions])


def evaluate_sample_wise(predictions, targets, verbose=True):
    auroc = roc_auc_score(targets, predictions)
    ap = average_precision_score(targets, predictions)

    if verbose:
        print(f"AUROC: {auroc:.4f}")
        print(f"AP: {ap:.4f}")

    return auroc, ap


def evaluate_pixel_wise(predictions, targets, ap=True, dice=False, proauc=False,
                        n_thresh_dice=10):
    if ap:
        ap = compute_average_precision(predictions, targets)
        print(f"AP: {ap:.4f}")
    else:
        ap = 0.0

    if dice:
        dice, th = compute_best_dice(
            predictions, targets, n_thresh=n_thresh_dice)
        print(f"DSC: {dice:.4f}, best threshold: {th:.4f}")
    else:
        dice = 0.0
        th = None

    if proauc:
        h, w = predictions.shape[-2:]
        compute_pro_auc(
            predictions=predictions.view(-1, 1, h, w),
            targets=targets.view(-1, 1, h, w),
        )

    return ap, dice, th


def full_evaluation_sample(predictions, targets, anomalies):
    """Perform the full sample-wise evaluation for every anomaly type

    Args:
        predictions (torch.Tensor): Predicted labels, shape [n]
        targets (torch.Tensor): Target labels, shape [n]
        anomalies (list of str): Anomaly types, len = n
    """
    unique_anomalies = set(anomalies)
    unique_anomalies.discard("normal")

    for anomaly in sorted(unique_anomalies):
        print(f"\nEvaluating performance on {anomaly}")

        # Filter only relevant anomalies (and "normal")
        considered = [anomaly, "normal"]
        p = torch.tensor(
            [m for m, a in zip(predictions, anomalies) if a in considered])
        t = torch.tensor(
            [l for l, a in zip(targets, anomalies) if a in considered])

        # Evaluate sample-wise
        evaluate_sample_wise(p, t, verbose=True)

    print("\nEvaluating total performance")
    evaluate_sample_wise(predictions, targets, verbose=True)


def full_evaluation_pixel(predictions, targets, anomalies):
    """Perform the full sample-wise evaluation for every anomaly type

    Args:
        predictions (torch.Tensor): Predicted anomaly maps, shape [n, slices, w, h]
        targets (torch.Tensor): Target segmentations, shape [n, slices, w, h]
        anomalies (list of str): Anomaly types, len = n
    """
    unique_anomalies = set(anomalies)
    unique_anomalies.discard("normal")

    for anomaly in sorted(unique_anomalies):
        print(f"\nEvaluating performance on {anomaly}")

        # Filter only relevant anomalies
        p = torch.cat(
            [m for m, a in zip(predictions, anomalies) if a == anomaly])
        t = torch.cat([l for l, a in zip(targets, anomalies) if a == anomaly])

        # Evaluate sample-wise
        evaluate_pixel_wise(p, t)

    print("\nEvaluating total performance")
    _, _, th = evaluate_pixel_wise(predictions, targets, dice=True)

    return th


def eval_dir(pred_dir, target_dir, mode):
    # List files from dir
    ext = "*.nii.gz" if mode == "pixel" else "*.txt"
    pred_files = glob(os.path.join(pred_dir, ext))
    target_files = glob(os.path.join(target_dir, ext))
    print(
        f"Found {len(pred_files)} predictions and {len(target_files)} targets.")
    assert len(pred_files) == len(target_files)

    # Get names of anomalies
    anomalies = [f.split('/')[-1].split(".nii.gz")[0][6:]
                 for f in target_files]
    print(f"Found anomalies: {set(anomalies)}")

    # Load from files
    if mode == "pixel":
        print("Reading nii files with predictions")
        preds = np.stack([data_utils.load_nii(f)[0] for f in pred_files])
        pred_img_size = preds[0].shape[-2]
        print("Reading nii files with targets")
        targets = np.stack([data_utils.load_nii(f, size=pred_img_size, dtype="short")[0]
                            for f in target_files])
        print(preds.shape, preds.dtype)
        print(targets.shape, targets.dtype)
        full_evaluation_pixel(predictions=torch.from_numpy(preds),
                              targets=torch.from_numpy(targets),
                              anomalies=anomalies)
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
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Directory with predicted anomaly maps or labels")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Directory with ground truth anomaly maps or labels")
    parser.add_argument("-m", "--mode", type=str, required=True,
                        help="Evaluation mode, choose between 'pixel' or 'sample'",
                        choices=["pixel", "sample"])
    args = parser.parse_args()

    eval_dir(args.input, args.output, args.mode)
