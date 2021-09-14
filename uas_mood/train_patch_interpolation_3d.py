import argparse
import gc
import os
import random
from time import time
from warnings import warn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from uas_mood.models import models
from uas_mood.utils import evaluation, utils
from uas_mood.utils.dataset import (
    MOODROOT,
    PatchSwapDataset3D,
    TestDataset3D,
    get_test_files,
    get_train_files,
)
from uas_mood.utils.hparam_search import hparam_search

# matplotlib can't be imported in a read-only filesystem
try:
    import matplotlib
    import matplotlib.pyplot as plt
except FileNotFoundError:
    pass


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
    def __init__(self, args):
        super().__init__()

        # Save all input args to self.hparams
        self.save_hyperparameters()
        self.args = self.hparams.args

        # Network
        self.net = models.ACSUNet(in_channels=1, out_channels=1, init_features=32)
        self.net.apply(models.weights_init_relu)

        # Example input array needed to log the graph in tensorboard
        input_size = (self.args.crop_size, self.args.crop_size, self.args.crop_size)
        self.example_input_array = torch.randn(
            [5, 1, *input_size])

        # Init Loss function
        self.loss_fn = torch.nn.BCELoss()

        if self.logger:
            self.logger.log_hyperparams(self.args)

    def forward(self, x):
        y = self.net(x)
        return y

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.args.lr, weight_decay=0.5 * 0.0005)
        return [optimizer]
        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': {
        #         'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
        #             optimizer, patience=2, factor=0.1),
        #         'monitor': 'loss',
        #         'interval': 'step',
        #         'frequency': 100,
        #     }
        # }

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
        # Reshape into slices, h, w
        inp = inp.view(-1, *inp.shape[-2:])
        anomaly_map = anomaly_map.view(-1, *anomaly_map.shape[-2:])
        target_seg = target_seg.view(-1, *target_seg.shape[-2:])
        # Shuffle
        perm = torch.randperm(len(inp))
        inp = inp[perm]
        anomaly_map = anomaly_map[perm]
        target_seg = target_seg[perm]

        # Prepare data structures
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
        x = x[0]
        y = y[0]
        pred = self(x)

        # Compute loss
        loss = self.loss_fn(pred, y)

        self.log("loss", loss.cpu(), on_step=True)

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
        x = x[0]
        y = y[0]
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
        ap = evaluation.compute_average_precision(
            out["anomaly_map"], torch.where(out["target_seg"] > 0, 1, 0))

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
        raise NotImplementedError

    def test_epoch_end(self, outputs):
        raise NotImplementedError


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
    train_ds = PatchSwapDataset3D(
        training_files, data=args.data, load_to_ram=args.load_to_ram,
        return_crops=True, batch_size=args.batch_size, crop_size=args.crop_size
    )
    trainloader = DataLoader(train_ds, shuffle=True, num_workers=16,
                             pin_memory=False, prefetch_factor=16)
    val_ds = PatchSwapDataset3D(
        val_files, data=args.data, load_to_ram=args.load_to_ram,
        return_crops=True, batch_size=args.batch_size, crop_size=args.crop_size
    )
    valloader = DataLoader(val_ds, shuffle=True, num_workers=16,
                           pin_memory=False, prefetch_factor=16)

    utils.printer(
        f"Finished loading training data in {time() - t_start:.2f}s", args.verbose)

    # Train
    model.start_time = time()
    utils.printer("Start training", args.verbose)
    trainer.fit(model, trainloader, valloader)

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
    ds = TestDataset3D(test_files, return_crops=True, crop_size=args.crop_size)
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
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_test_files", type=int, default=None)
    parser.add_argument("--val_every_epoch", type=float, default=1/3)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--no_load_to_ram", dest="load_to_ram",
                        action="store_false")
    # Data params
    parser.add_argument("--data", type=str, default="brain",
                        choices=["brain", "abdom"])
    parser.add_argument("--crop_size", type=int, default=64)
    # Engineering params
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    # Logging params
    parser.add_argument("--log_dir", type=str,
                        default=f"{os.path.dirname(os.path.abspath(__file__))}/logs/")
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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Handle ~ in data_paths
    args.log_dir = os.path.expanduser(args.log_dir)

    print(args.log_dir)

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

    # Get train and test paths
    train_files = get_train_files(MOODROOT, args.data)
    test_files = get_test_files(MOODROOT, args.data)

    if args.max_train_files is not None:
        train_files = train_files[:args.max_train_files]
    if args.max_test_files is not None:
        test_files = test_files[:args.max_test_files]

    if args.debug:
        pass
        # train_files = train_files[:40]
        # test_files = test_files[:10]

    callbacks = [LitProgressBar()]

    # Init logger
    if args.debug:
        logger = None
    else:
        logger = TensorBoardLogger(
            args.log_dir, name="patch_interpolation_3d", log_graph=True)
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
