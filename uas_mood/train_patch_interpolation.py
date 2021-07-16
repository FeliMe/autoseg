import argparse
import gc
import os
import random
from time import time
from warnings import warn

import matplotlib
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from uas_mood.models import models
from uas_mood.utils import evaluation, utils
from uas_mood.utils.data_utils import save_nii, volume_viewer
from uas_mood.utils.dataset import (
    MOODROOT,
    PatchSwapDataset,
    TestDataset,
    get_test_files,
    get_train_files,
)
from uas_mood.utils.hparam_search import hparam_search


class LitProgressBar(pl.callbacks.progress.ProgressBar):
    def init_validation_tqdm(self):
        return tqdm(disable=True)


def plot(images):
    if not isinstance(images, list):
        images = [images]
    n = len(images)
    fig = plt.figure(figsize=(4 * n, 4))
    plt.axis("off")
    for i, im in enumerate(images):
        fig.add_subplot(1, n, i + 1)
        plt.imshow(im, cmap="gray", vmin=0., vmax=1.)
    plt.show()


def plot_volume(volumes, i_slice):
    if not isinstance(volumes, list):
        volumes = [volumes]
    for vol in volumes:
        assert vol.ndim == 3
    n = len(volumes)
    fig = plt.figure(figsize=(4 * n, 4))
    plt.axis("off")
    for i, vol in enumerate(volumes):
        fig.add_subplot(1, n, i + 1)
        plt.imshow(vol[i_slice], cmap="gray", vmin=0., vmax=1.)
    plt.show()


class LitModel(pl.LightningModule):
    def __init__(self, args=None):
        super().__init__()

        # Save all args
        self.args = args

        # Network
        if args.model == "unet":
            print("Using UNet")
            self.net = torch.hub.load('mateuszbuda/brain-segmentation-pytorch',
                                      'unet', in_channels=1, out_channels=1,
                                      init_features=32, pretrained=False,
                                      verbose=False)
            self.net.apply(models.weights_init_relu)
        else:
            print("Using Wide ResNet")
            self.net = models.WideResNetAE(inp_size=args.img_size,
                                           widen_factor=args.model_width)

        # Example input array needed to log the graph in tensorboard
        input_size = (1, args.img_size, args.img_size)
        self.example_input_array = torch.randn(
            [5, *input_size])

        # Print model summary
        # from torchsummary import summary
        # print(summary(self.net, input_size=input_size, device="cpu"))
        # import IPython
        # IPython.embed()
        # exit(1)

        # Init Loss function
        self.loss_fn = torch.nn.BCELoss()

        if self.logger:
            self.logger.log_hyperparams(self.args)
            self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.args.lr, weight_decay=0.5 * 0.0005)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9995)
        # return [opt], [scheduler]
        return [opt]

    def print_(self, msg):
        if self.args.verbose:
            print(msg)

    def log_metric(self, name, value, on_step=None, on_epoch=None):
        if self.logger:
            self.log(name, value, on_step=on_step, on_epoch=on_epoch,
                     logger=True)
        if self.args.hparam_search:
            tune.report(value)

    @staticmethod
    def stack_outputs(outputs):
        """concatenates a list of of dictionaries with torch.tensors to a
        single dict"""
        out = {}
        for key in outputs[0].keys():
            # First stack all non-tensor types to a list
            if not isinstance(outputs[0][key], torch.Tensor):
                out[key] = [o[key] for o in outputs]
            elif outputs[0][key].ndim == 0:
                out[key] = torch.stack([o[key] for o in outputs], 0)
            else:
                out[key] = torch.cat([o[key] for o in outputs], 0)
        return out

    def plot_reconstruction(self, inp, anomaly_map, target_seg):
        images = [
            inp.cpu(),
            anomaly_map.cpu(),
            target_seg.cpu(),
        ]
        titles = [
            "Input",
            "Anomaly map",
            "Target segmentation",
        ]

        fig = evaluation.plot_results(
            images=images,
            titles=titles,
            n_images=self.args.num_images_log
        )

        # Return
        return fig

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        # Compute loss
        loss = self.loss_fn(pred, y)

        return {"loss": loss.cpu()}

    def training_epoch_end(self, outputs):
        # Stack all values to a dict
        out = self.stack_outputs(outputs)

        # Print training epoch summary
        self.print_(
            f"Epoch [{self.current_epoch}/{self.args.max_epochs}] Train loss: {out['loss'].mean():.4f}")

        # Tensorboard logs
        self.log_metric("train_loss", out["loss"].mean())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        # Compute loss
        loss = self.loss_fn(pred, y)

        return {
            "inp": x.cpu(),
            "target_seg": y.cpu(),
            "loss": loss.cpu(),
            "anomaly_map": pred.cpu(),
        }

    def validation_epoch_end(self, outputs):
        self.print_("Validating")
        val_start = time()
        # Stack all values to a dict
        out = self.stack_outputs(outputs)

        # Compute average precision
        # Only use half of the total validation samples to reduce time
        val_samples = len(out["anomaly_map"])
        ap = evaluation.compute_average_precision(
            out["anomaly_map"][:val_samples // 3], torch.where(out["target_seg"] > 0, 1, 0)[:val_samples // 3])

        self.log_metric("ap", ap)
        self.log_metric("val_loss", out["loss"].mean())

        # Tensorboard logs
        if self.logger:
            # log a reconstructed sample
            self.print_("Logging a validation sample to tensorboard")
            fig = self.plot_reconstruction(
                out["inp"], out["anomaly_map"], out["target_seg"])
            tb = self.logger.experiment
            tb.add_figure("val sample", fig, global_step=self.global_step)

        # Log to tune if hyperparameter search
        if self.args.hparam_search:
            tune.report(ap=ap)

        # Get epoch timings
        time_elapsed, time_per_epoch, time_left = utils.get_training_timings(
            self.start_time, self.current_epoch * args.val_every_epoch, self.args.max_epochs
        )
        # Print validation summary
        self.print_(
            f"Pred max: {out['anomaly_map'].max():.4f}, pred min: {out['anomaly_map'].min():.4f}")
        self.print_(f"Val loss: {out['loss'].mean():.4f}, "
                    f"average precision val: {ap:.4f}\n"
                    f"Time elapsed: {time_elapsed}, "
                    f"Time left: {time_left}, "
                    f"Time per epoch: {time_per_epoch}, "
                    f"Time for validation: {time() - val_start:.2f}s")

    def test_step(self, batch, batch_idx):
        x, y = batch

        # Save filename for evaluation
        name = x[0]
        anomaly = name.split('/')[-1].split(".nii.gz")[0][6:]

        # x was a tuple of (file_name, volume [slices, w, h])
        x = x[1]
        # y was a tuple of (file_name, volume [slices, w, h])
        y = y[1].cpu()

        # Forward from all three viewing directions
        pred_axial = self(x.unsqueeze(1)).squeeze(1).cpu()
        # Equivalent to np.rollaxis(x, 1)
        pred_coronal = self(x.permute(1, 0, 2).unsqueeze(1)).squeeze(1).cpu()
        pred_coronal = pred_coronal.permute(1, 0, 2)  # Roll back
        # Equivalent to np.rollaxis(x, 2)
        pred_saggital = self(x.permute(2, 0, 1).unsqueeze(1)).squeeze(1).cpu()
        pred_saggital = pred_saggital.permute(1, 2, 0)  # Roll back

        # Combine viewing directions
        pred = torch.stack([pred_axial, pred_coronal, pred_saggital]).mean(0)

        # Compute loss
        loss = self.loss_fn(pred, y.float())

        return {
            "inp": x.cpu(),
            "target_seg": y.cpu(),
            "name": name,
            "anomaly": anomaly,
            "loss": loss.cpu(),
            "anomaly_map": pred,
        }

    def test_epoch_end(self, outputs):
        # Stack all values to a dict
        out = self.stack_outputs(outputs)
        print(f"Test loss: {out['loss'].mean():.4f}")

        inp = out["inp"]
        target_seg = out["target_seg"]
        pred = out["anomaly_map"]
        anomalies = out["anomaly"]
        print(pred.min(), pred.max(), pred.mean())

        # Separate predictions and labels into complete scans
        inp = inp.reshape(-1, *self.args.volume_shape)
        pred = pred.reshape(-1, *self.args.volume_shape)
        target_seg = target_seg.reshape(-1, *self.args.volume_shape)

        # Get labels from segmentation masks and anomaly maps
        target_label = torch.where(target_seg.sum((1, 2, 3)) > 0, 1, 0)
        pred_label = evaluation.fpi_sample_score_list(predictions=pred)

        # Perform evaluation for all anomalies separately
        print("----- SAMPLE-WISE EVALUATION -----")
        evaluation.full_evaluation_sample(pred_label, target_label, anomalies)
        print("\n----- PIXEL-WISE EVALUATION -----")
        th = evaluation.full_evaluation_pixel(pred, target_seg, anomalies)

        if self.logger:
            # Log to tensorboard
            tb = self.logger.experiment

            # Binarize map
            bin_map = torch.where(pred > th, 1., 0.)

            unique_anomalies = set(anomalies)
            unique_anomalies.discard("normal")
            for anomaly in unique_anomalies:
                print(
                    f"Writing test sample images of {anomaly} to tensorboard")
                # Filter only relevant anomalies
                x = torch.cat(
                    [m for m, a in zip(inp, anomalies) if a == anomaly])
                p = torch.cat(
                    [m for m, a in zip(pred, anomalies) if a == anomaly])
                b = torch.cat(
                    [m for m, a in zip(bin_map, anomalies) if a == anomaly])
                t = torch.cat(
                    [m for m, a in zip(target_seg, anomalies) if a == anomaly])

                # Shuffle before plotting
                perm = torch.randperm(len(x))
                x = x[perm]
                p = p[perm]
                b = b[perm]
                t = t[perm]

                has_anomaly = torch.where(t.sum((1, 2)) > 0, True, False)

                images = [
                    x[has_anomaly].unsqueeze(1),
                    p[has_anomaly].unsqueeze(1),
                    b[has_anomaly].unsqueeze(1),
                    t[has_anomaly].unsqueeze(1),
                ]
                titles = [
                    "Input image",
                    "Anomaly map",
                    "Binarized map",
                    "Ground turth",
                ]

                # Log sample images to tensorboard
                fig = evaluation.plot_results(
                    images=images,
                    titles=titles,
                    n_images=10
                )
                tb.add_figure(
                    f"Test samples {anomaly}", fig, global_step=self.global_step)

            # Log precision-recall curve to tensorboard
            fig = evaluation.plot_prc(
                predictions=out["anomaly_map"],
                targets=target_seg
            )
            tb.add_figure("Precision-Recall Curve", fig,
                          global_step=self.global_step)


def train(args, trainer, train_files):
    # Init lighning model
    if args.model_ckpt:
        utils.printer(
            f"Restoring checkpoint from {args.model_ckpt}", args.verbose)
        model = LitModel.load_from_checkpoint(args.model_ckpt, args=args)
    else:
        model = LitModel(args)

    # Tkinter backend fails on server, switching to Agg
    if not args.debug:
        matplotlib.use('Agg')

    # Load data
    utils.printer("Loading training data", args.verbose)
    t_start = time()

    # Check if enough RAM available
    if args.load_to_ram:
        utils.check_ram(train_files)

    # Split into train- and val-files
    random.shuffle(train_files)
    split_idx = int((1 - args.val_fraction) * len(train_files))
    training_files = train_files[:split_idx]
    val_files = train_files[split_idx:]
    utils.printer(f"Training with {len(training_files)} files", args.verbose)
    utils.printer(f"Validating on {len(val_files)} files", args.verbose)

    # Create Datasets and Dataloaders
    train_ds = PatchSwapDataset(training_files, args.img_size,
                                args.slices_lower_upper, data=args.data)
    trainloader = DataLoader(train_ds, batch_size=args.batch_size,
                             num_workers=args.num_workers, shuffle=True)
    val_ds = PatchSwapDataset(val_files, args.img_size,
                              args.slices_lower_upper, data=args.data)
    valloader = DataLoader(val_ds, batch_size=args.batch_size,
                           num_workers=args.num_workers, shuffle=True)

    utils.printer(
        f"Finished loading training data in {time() - t_start:.2f}s", args.verbose)

    # Train
    model.start_time = time()
    utils.printer("Start training", args.verbose)
    trainer.fit(model, trainloader, valloader)

    # Delete train data to free memory
    trainer.train_dataloader = None
    trainer.val_dataloaders = None
    del train_ds, trainloader, val_ds, valloader
    gc.collect()

    # Return the trained model
    return model


def test(args, trainer, test_files, model=None):
    # Init lighning model
    if args.model_ckpt is None and model is None:
        warn("Evaluating untrained model")
        model = LitModel(args)
    elif model is None:
        print(f"Restoring checkpoint from {args.model_ckpt}")
        model = LitModel.load_from_checkpoint(args.model_ckpt, args=args)

    # Check if enough RAM available
    if args.load_to_ram:
        utils.check_ram(test_files)

    # Load data
    print("Loading data")
    t_start = time()
    ds = TestDataset(test_files, args.img_size, args.slices_lower_upper)
    print(f"Finished loading data in {time() - t_start:.2f}s")

    # Test
    print("Testing model")
    trainer.test(model, ds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # General script control params
    parser.add_argument("--no_train", dest="train", action="store_false")
    parser.add_argument("--no_test", dest="test", action="store_false")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--model_ckpt", type=str, default=None)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--val_every_epoch", type=float, default=1/3)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--no_load_to_ram", dest="load_to_ram",
                        action="store_false")
    # Data params
    parser.add_argument("--data", type=str, default="brain",
                        choices=["brain", "abdom"])
    parser.add_argument("--img_size", type=int, default=256)
    # parser.add_argument('--slices_lower_upper',
    #                     nargs='+', type=int, default=[23, 200])
    # parser.add_argument('--slices_lower_upper',
    #                     nargs='+', type=int, default=[79, 83])
    # Engineering params
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    # Logging params
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--num_images_log", type=int, default=4)
    # Ray tune params
    parser.add_argument("--hparam_search", action="store_true")
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--cpu_per_trial", type=int, default=8)
    parser.add_argument("--gpu_per_trial", type=float, default=0.25)
    parser.add_argument("--target_metric", type=str, default="ap")
    # Model params
    parser.add_argument("--model", type=str, choices=["unet", "resnet"],
                        default="unet")
    parser.add_argument("--model_width", type=int, default=4)
    # Real Hyperparameters
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Handle ~ in data_paths
    args.log_dir = os.path.expanduser(args.log_dir)

    # Check if GPU is available
    if not torch.cuda.is_available():
        args.gpus = 0
        args.num_workers = 0

    # Reproducibility
    pl.seed_everything(args.seed)

    # Handle fractional validations
    if args.val_every_epoch < 1:
        check_val_every_n_epoch = 1
        val_check_interval = args.val_every_epoch
    else:
        check_val_every_n_epoch = args.val_every_epoch
        val_check_interval = 1

    # Save number of slices per sample as a parameters
    # args.n_slices = args.slices_lower_upper[1] - args.slices_lower_upper[0]
    args.slices_lower_upper = None
    args.n_slices = args.img_size
    args.volume_shape = [args.n_slices, args.img_size, args.img_size]

    # Get train and test paths
    train_files = get_train_files(MOODROOT, args.data)
    test_files = get_test_files(MOODROOT, args.data)
    # train_files = train_files[:50]
    # test_files = test_files[:10]

    callbacks = [LitProgressBar()]

    # Init logger
    if args.debug:
        logger = None
    else:
        logger = TensorBoardLogger(
            args.log_dir, name="patch_interpolation", log_graph=True)
        # Add a ModelCheckpoint callback. Always log last ckpt and best train
        callbacks += [ModelCheckpoint(monitor=args.target_metric, mode='max',
                                      save_last=True)]

    # Init trainer
    trainer = pl.Trainer(gpus=args.gpus,
                         callbacks=callbacks,
                         logger=logger,
                         precision=args.precision,
                         progress_bar_refresh_rate=100,
                         checkpoint_callback=not args.debug,
                         check_val_every_n_epoch=check_val_every_n_epoch,
                         val_check_interval=val_check_interval,
                         num_sanity_val_steps=0,
                         # min_epochs=args.max_epochs,
                         max_epochs=args.max_epochs)

    if args.hparam_search:
        # Hyperparameter search with ray tune
        search_config = {
            "lr": tune.loguniform(1e-5, 1e-4),
            # "batch_size": tune.choice([32, 64, 128]),
            "model_width": tune.choice([2, 3, 4, 5]),
        }
        hparam_search(
            search_config=search_config,
            args=args,
            trainer=trainer,
            train_files=train_files,
            train_fn=train
        )
    else:
        model = None
        if args.train:
            model = train(args, trainer, train_files)
        if args.test:
            test(args, trainer, test_files, model=model)
