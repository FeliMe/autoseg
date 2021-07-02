"""
My implementation of: "PaDiM: a Patch Distribution Modeling Framework
for Anomaly Detection and Localization"

https://arxiv.org/pdf/2011.08785.pdf
"""
import argparse
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm

import uas_mood.models.feature_extractor as feature
from uas_mood.utils import evaluation, utils
from uas_mood.utils.data_utils import volume_viewer
from uas_mood.utils.dataset import (
    MOODROOT,
    TestDataset,
    TrainDataset,
    get_test_files,
    get_train_files,
)


def partial_train(args, loader, extractor, slices):
    """Compute mean and covariance for a range of slices

    Args:
        loader (torch.utils.data.DataLoader): Must be a sequential loader 
                                              without shuffling
        extractor (nn.Module): Feature extractor for the single slices
        slices (slice): slice range

    Returns:
        mu (torch.Tensor): mean per voxel
        cov_inv (torch.Tensor): Inverse of the covariance matrices per voxel
    """
    assert not loader.drop_last
    assert isinstance(loader.sampler, SequentialSampler)

    feats = []
    for sample in loader:
        with torch.no_grad():
            # Samples are [slices, 1, w, h]
            sample = sample[slices].to(args.device)
            # Feats are [slices, f, w, h]
            feat = extractor(sample).cpu()
            # Convert to [f, slices, w, h]
            feat = feat.transpose(0, 1)
            feats.append(feat)

    # Stack to [n, f, slices, w, h]
    feats = torch.stack(feats, dim=0)

    n, f, n_slices, w, h = feats.shape
    d = n_slices * w * h
    feats = feats.reshape(n, f, d)

    # Compute mean
    mu = feats.mean(dim=0).T  # [d, f]
    # mu = mu.half()

    # Covariance
    feats = feats.numpy()
    feats = np.transpose(feats, (2, 0, 1))  # [d, n, f]
    cov = [np.cov(feat, rowvar=False) + (args.eps * np.eye(f))
           for feat in feats]  # [d, f, f]

    # Inverting covariance
    cov_inv = [np.linalg.inv(c) for c in cov]  # [d, f, f]
    cov_inv = np.stack(cov_inv, axis=0)  # [d, f, f]
    cov_inv = torch.from_numpy(cov_inv)  # [d, f, f]
    # cov_inv = cov_inv.half()

    # Reshape mu back to [slices, w, h, f]
    mu = mu.reshape(n_slices, w, h, f)
    # Reshape cov_inv back to [slices, w, h, f, f]
    cov_inv = cov_inv.reshape(n_slices, w, h, f, f)

    return mu, cov_inv


def train(args, extractor, train_files):
    print("TRAINING")
    t_start = time()
    print("Loading data")
    ds = TrainDataset(train_files, args.img_size, args.slices_lower_upper)
    loader = DataLoader(ds, batch_size=args.n_slices, num_workers=0)
    print(f"Finished loading data in {time() - t_start:.2f}s")
    mu, cov_inv = [], []
    for lower in tqdm(range(0, args.n_slices, args.n_slices_batch)):
        upper = lower + args.n_slices_batch
        slices = slice(lower, upper)
        mu_slice, cov_inv_slice = partial_train(args, loader, extractor, slices)
        mu.append(mu_slice)
        cov_inv.append(cov_inv_slice)

    mu = torch.cat(mu, dim=0)  # [slice, w, h, f]
    cov_inv = torch.cat(cov_inv, dim=0)  # [slice, w, h, f, f]

    # Save results
    results = {'mu': mu, 'cov_inv': cov_inv}
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Saving parameters to {args.save_path}")
    torch.save(results, args.save_path)


def test(args, extractor, test_files):
    print("TESTING")

    # Loading results
    print("Loading mu and cov")
    results = torch.load(args.save_path)
    mu = results['mu']  # [slice, w_, h_, f]
    cov_inv = results['cov_inv']  # [slice, w_, h_, f, f]

    # Reshape mu and cov_inv
    slices, w_, h_, f = mu.shape
    d = slices * w_ * h_
    mu = mu.reshape(d, f)  # [d, f]
    cov_inv = cov_inv.reshape(d, f, f)  # [d, f, f]

    # Loading data
    print("Loading data")
    t_start = time()
    ds = TestDataset(test_files, args.img_size, args.slices_lower_upper)
    print(f"Finished loading data in {time() - t_start:.2f}s")

    anomaly_maps = []
    segmentations = []
    samples = []
    anomalies = []
    for batch in tqdm(ds):
        sample, seg = batch
        name = sample[0]
        anomaly = name.split('/')[-1].split(".nii.gz")[0][6:]
        sample = sample[-1]  # sample is tuple (path, volume), volume [slices, w, h]
        seg = seg[-1]  # seg is tuple (path, volume), volume [slices, w, h]

        with torch.no_grad():
            sample = sample.unsqueeze(1).to(args.device)  # [slices, 1, w, h]
            feat = extractor(sample).cpu().transpose(0, 1)  # [f, slices, w_, h_]

            # Reshape feature map
            f, slices, w_, h_ = feat.shape
            d = slices * w_ * h_
            feat = feat.reshape(f, d)  # [f, d]
            feat = feat.T  # [d, f]

            # Compute anomaly map with mahalanobis distance
            anomaly_map = [torch.sqrt(((x - m).T @ c_inv.float() @ (x - m)))
                           for x, m, c_inv in zip(feat, mu, cov_inv)]
            anomaly_map = torch.stack(anomaly_map, dim=0)  # [d, 1]
            anomaly_map = anomaly_map.reshape(1, slices, w_, h_)

            # Upsample anomaly map
            anomaly_map = F.interpolate(anomaly_map, size=args.img_size,
                                        mode="bilinear", align_corners=True)

            # Post processing
            blur = utils.GaussianSmoothing(channels=1,
                                           kernel_size=5, sigma=4, dim=3)
            anomaly_map = blur(anomaly_map[None]).squeeze(0)

        anomaly_maps.append(anomaly_map)
        segmentations.append(seg)
        anomalies.append(anomaly)
        if len(samples) <= args.n_imgs_log:
            samples.append(sample.transpose(0, 1).cpu())  # [1, slices, w, h]

    anomaly_maps = torch.cat(anomaly_maps, dim=0)  # [n, slices, w, h]
    segmentations = torch.stack(segmentations, dim=0)  # [n, slices, w, h]
    samples = torch.cat(samples, dim=0)  # [n_imgs_log, slices, w, h]

    unique_anomalies = set(anomalies)
    # We can't evaluate localization on normal samples
    unique_anomalies.discard("normal")

    for anomaly in sorted(unique_anomalies):
        print(f"\nEvaluating performance on {anomaly}")
        predictions = torch.cat([m for m, a in zip(anomaly_maps, anomalies) if a == anomaly], dim=0)
        targets = torch.cat([s for s, a in zip(segmentations, anomalies) if a == anomaly], dim=0)
        evaluation.evaluate(
            predictions=predictions,
            targets=targets,
        )

    print("\nEvaluating total performance")
    _, _, th = evaluation.evaluate_pixel_wise(
        predictions=anomaly_maps,
        targets=segmentations,
    )

    # Binarize anomaly_maps
    bin_map = torch.where(anomaly_maps > th, 1., 0.)

    print("Saving some images")
    c = args.n_slices // 2
    images = [
        samples[:, c][:, None],
        anomaly_maps[:, c][:, None],
        bin_map[:, c][:, None],
        segmentations[:, c][:, None]
    ]
    titles = ['Input', 'Anomaly map', 'Binarized map', 'Ground truth']
    evaluation.plot_results(images, titles, n_images=args.n_imgs_log)
    plt.savefig(f"{args.save_dir}/samples.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # General script params
    parser.add_argument("--no_train", dest="train", action="store_false")
    parser.add_argument("--no_test", dest="test", action="store_false")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_slices_batch", type=int, default=4)
    # Data params
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--ds", type=str, default="brain",
                        choices=["brain", "abdom"])
    # parser.add_argument('--slices_lower_upper',
    #                      nargs='+', type=int, default=[23, 200])
    parser.add_argument('--slices_lower_upper',
                         nargs='+', type=int, default=[127, 131])
    # Feature extractor params
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=feature.Extractor.BACKBONENETS)
    parser.add_argument("--cnn_layers", type=str,
                        default=["layer1", "layer2", "layer3"])
    parser.add_argument("--eps", type=float, default=0.01)
    # Logging params
    parser.add_argument("--save_dir", type=str, default='./logs/padim/')
    parser.add_argument("--n_imgs_log", type=int, default=10)
    # Other
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Random seed
    pl.seed_everything(args.seed)

    # Select device
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Save path
    args.save_dir = f"{args.save_dir}{args.ds}_{args.backbone}_{args.img_size}" \
                    f"_{args.slices_lower_upper[0]}-{args.slices_lower_upper[1]}"
    args.save_path = f"{args.save_dir}/results.pt"

    # Save number of slices per sample as a parameters
    args.n_slices = args.slices_lower_upper[1] - args.slices_lower_upper[0]

    # Get train and test paths
    train_files = get_train_files(MOODROOT, args.ds)
    test_files = get_test_files(MOODROOT, args.ds)
    # test_files = test_files[:30]  # TODO: remove

    # Prepare feature extractor
    extractor = feature.Extractor(
        backbone=args.backbone,
        cnn_layers=args.cnn_layers,
        img_size=args.img_size,
        upsample="nearest",
        is_agg=False,
        featmap_size=args.img_size // 4,
        keep_feature_prop=(200 / 448)
    ).to(args.device)

    if args.train:
        train(args, extractor, train_files)

    if args.test:
        test(args, extractor, test_files)
