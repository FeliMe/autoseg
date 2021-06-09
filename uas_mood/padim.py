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
from tqdm import tqdm

import uas_mood.models.feature_extractor as feature
from uas_mood.utils.dataset import (
    MOODROOT,
    TestDataset,
    TrainDataset,
    get_test_files,
    get_train_files,
)


def train(args, extractor, train_files):
    print("TRAINING")
    feats = []
    print("Loading data")
    t_start = time()
    ds = TrainDataset(train_files, args.img_size, args.slices_lower_upper)
    loader = DataLoader(ds, batch_size=args.n_slices, num_workers=8)
    print(f"Finished loading data in {time() - t_start:.2f}s")
    for sample in tqdm(loader):
        import IPython ; IPython.embed() ; exit(1)
        with torch.no_grad():
            # Processing all slices is not feasible, therefore split them
            batches = sample.split(args.batch_size, dim=0)
            feat_batch = []
            for x in batches:
                # x are [batch, 1, w, h]
                x = x.to(args.device)
                # feats are [batch, f, w, h]
                feat = extractor(x).cpu()
                feat_batch.append(feat)

            # Concatenate to [slices, f, w, h]
            feat_batch = torch.cat(feat_batch, dim=0)
            # Get features first
            feat_batch = feat_batch.transpose(0, 1)
        feats.append(feat_batch)

    # Stack to [n, slices, f, w, h]
    feats = torch.stack(feats, dim=0)
    n, f, slices, h, w = feats.shape
    d = slices * h * w
    feats = feats.reshape(n, f, d)

    # Compute feature statistics for every slice and patch
    print("Computing mean")
    mu = feats.mean(dim=0).T  # [d, f]

    # Covariance
    feats = feats.numpy()
    feats = np.transpose(feats, (2, 0, 1))  # [d, n, f]
    print("Computing cov")
    cov = [np.cov(feat, rowvar=False) + (args.eps * np.eye(f))
           for feat in feats]  # [d, f, f]
    print("Inverting cov")
    cov_inv = [np.linalg.inv(c) for c in cov]  # [d, f, f]
    cov_inv = np.stack(cov_inv, axis=0)  # [d, f, f]
    cov_inv = torch.from_numpy(cov_inv)  # [d, f, f]
    cov = np.stack(cov, axis=0)  # [d, f, f]
    cov = torch.from_numpy(cov)  # [d, f, f]

    # Save results
    results = {'mu': mu, 'cov': cov, 'cov_inv': cov_inv.float()}
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Saving parameters to {args.save_path}")
    torch.save(results, args.save_path)


def test(args, extractor, test_files):
    print("TESTING")

    # Loading results
    results = torch.load(args.save_path)
    mu = results['mu']  # [d, f]
    cov_inv = results['cov_inv']  # [d, f, f]

    # Loading data
    print("Loading data")
    t_start = time()
    ds = TestDataset(test_files, args.img_size)
    print(f"Finished loading data in {time() - t_start:.2f}s")

    samples = []
    anomaly_maps = []
    segmentations = []
    masks = []
    for batch in tqdm(ds):
        sample, seg, mask = batch

        with torch.no_grad():
            sample = sample.transpose(0, 1).to(args.device)
            feat = extractor(sample).cpu().transpose(0, 1)

            # Reshape feature map
            f, slices, h, w = feat.shape
            d = slices * h * w
            feat = feat.reshape(f, d)  # [f, d]
            feat = feat.T  # [d, f]

            # Compute anomaly map with mahalanobis distance
            anomaly_map = [torch.sqrt(((x - m).T @ c_inv @ (x - m)))
                           for x, m, c_inv in zip(feat, mu, cov_inv)]
            # anomaly_map = [((x - m).T @ c_inv @ (x - m)) + c_logdet
            #                for x, m, c_inv, c_logdet in zip(feat, mu, cov_inv, cov_logdet)]
            anomaly_map = torch.stack(anomaly_map, dim=0)  # [d, 1]
            anomaly_map = anomaly_map.reshape(1, slices, h, w)

            # Upsample anomaly map
            anomaly_map = F.interpolate(anomaly_map, size=args.img_size,
                                        mode="bilinear", align_corners=True)

            # Post processing
            blur = utils.GaussianSmoothing(channels=1,
                                           kernel_size=5, sigma=4, dim=3)
            anomaly_map = blur(anomaly_map[None]).squeeze(0)
            # blur = utils.MedianFilter2d(kernel_size=5)
            # anomaly_map = blur(anomaly_map)

        anomaly_maps.append(anomaly_map)
        segmentations.append(seg)
        masks.append(mask)
        if len(samples) <= args.n_imgs_log:  # Save some memory
            samples.append(sample.transpose(0, 1).cpu())

    anomaly_maps = torch.cat(anomaly_maps, dim=0)
    segmentations = torch.cat(segmentations, dim=0)
    masks = torch.cat(masks, dim=0)
    samples = torch.cat(samples, dim=0)

    _, _, _, th = evaluation.evaluate(
        predictions=anomaly_maps,
        targets=segmentations,
        masks=masks
    )

    # Binarize anomaly_maps
    bin_map = torch.where(anomaly_maps > th, 1., 0.)

    print("Saving some images")
    c = (args.slices_lower_upper[1] - args.slices_lower_upper[0]) // 2
    images = [
        samples[:, c][:, None],
        anomaly_maps[:, c][:, None],
        bin_map[:, c][:, None],
        segmentations[:, c][:, None]
    ]
    titles = ['Input', 'Anomaly map', 'Binarized map', 'Ground truth']
    fig = evaluation.plot_results(images, titles, n_images=args.n_imgs_log)
    plt.savefig(f"{args.save_dir}{args.test_ds}_samples.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # General script params
    parser.add_argument("--no_train", dest="train", action="store_false")
    parser.add_argument("--no_test", dest="test", action="store_false")
    parser.add_argument("--batch_size", type=int, default=128)
    # Data params
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--ds", type=str, default="brain",
                        choices=["brain", "abdom"])
    parser.add_argument('--slices_lower_upper',
                         nargs='+', type=int, default=[35, 226])
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
    args.save_dir = f"{args.save_dir}{args.ds}_{args.backbone}_{args.img_size}"
    args.save_path = f"{args.save_dir}results.pt"

    # Save number of slices per sample as a parameters
    args.n_slices = args.slices_lower_upper[1] - args.slices_lower_upper[0]

    # Get train and test paths
    train_files = get_train_files(MOODROOT, args.ds)[:10]
    val_files = get_test_files(MOODROOT, args.ds, "val")

    # Prepare feature extractor
    extractor = feature.Extractor(
        backbone=args.backbone,
        cnn_layers=args.cnn_layers,
        img_size=args.img_size,
        upsample="nearest",
        featmap_size=args.img_size,
        keep_feature_prop=(200 / 448)
    ).to(args.device)

    if args.train:
        train(args, extractor, train_files)

    if args.test:
        test(args, extractor, val_files)
