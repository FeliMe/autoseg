import argparse
import os
import random
from time import time

import matplotlib
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
import torch
from torch.utils.data import DataLoader

from uas_mood import pytorch_ssim
from uas_mood.models.fae import FeatureAE
from uas_mood.utils import evaluation, utils
from uas_mood.utils.data_utils import volume_viewer
from uas_mood.utils.dataset import (
    MOODROOT,
    PatchSwapDataset,
    TestDataset,
    TrainDataset,
    get_test_files,
    get_train_files,
)
from uas_mood.utils.hparam_search import hparam_search


class LitModel(pl.LightningModule):
    def __init__(self, args=None):
        super().__init__()

        # Save all args
        self.args = args

        # Network
        self.net = FeatureAE(
            img_size=args.img_size,
            c_z=args.z_dim,
            ks=args.kernel_size,
            use_batchnorm=True,
        )

        # Example input array needed to log the graph in tensorboard
        self.example_input_array = torch.randn(
            [5, 1, args.img_size, args.img_size])

        # Select loss function
        print(f"Using {args.loss_fn} as a loss function")
        if args.loss_fn == "L2":
            self.loss_fn = torch.nn.MSELoss(reduction='mean')
            self.anomaly_fn = torch.nn.MSELoss(reduction='none')
        elif args.loss_fn == "L1":
            self.loss_fn = torch.nn.L1Loss(reduction='mean')
            self.anomaly_fn = torch.nn.L1Loss(reduction='none')
        elif args.loss_fn == "SSIM":
            self.loss_fn = pytorch_ssim.SSIMLoss(size_average=True)
            self.anomaly_fn = pytorch_ssim.SSIMLoss(size_average=False)
        else:
            raise NotImplementedError(f"{args.loss_fn} is not implemented")

        if self.logger:
            self.logger.log_hyperparams(self.args)
            self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9995)
        # return [opt], [scheduler]
        return [opt]

    def print_(self, msg):
        if self.args.verbose:
            print(msg)

    def log_metric(self, name, value):
        if self.logger:
            self.log(name, value, logger=True)
        if self.args.hparam_search:
            tune.report(value)

    def compute_anomaly_map(self, rec, feats):
        anomaly_map = torch.mean(self.anomaly_fn(rec, feats),
                                 dim=1, keepdim=True)
        anomaly_map = F.interpolate(anomaly_map, size=self.args.img_size,
                                    mode="bilinear", align_corners=True)
        return anomaly_map

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

    def training_step(self, x, batch_idx):
        # Forward slice
        feats, feats_rec = self(x)

        # Compute loss
        loss = self.loss_fn(feats_rec, feats).mean()

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        # Stack all values to a dict
        out = self.stack_outputs(outputs)

        # Print training epoch summary
        self.print_(f"Epoch [{self.current_epoch + 1}/{self.args.max_epochs}] Train - loss: {out['loss'].mean():.4f}")

        # Tensorboard logs
        self.log_metric("train_loss", out["loss"].mean())

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Forward slice
        feats, feats_rec = self(x)

        # Compute loss
        loss = self.loss_fn(feats_rec, feats).mean()

        # Get an anomaly map
        anomaly_map = self.compute_anomaly_map(feats_rec, feats)

        return {
            "inp":  x.cpu(),
            "target_seg": y.cpu(),
            "loss": loss,
            "anomaly_map": anomaly_map.cpu(),
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
            fig = self.plot_reconstruction(
                out["inp"], out["anomaly_map"], out["target_seg"])
            tb = self.logger.experiment
            tb.add_figure("val sample", fig, global_step=self.global_step)

        # Log to tune if hyperparameter search
        if self.args.hparam_search:
            tune.report(ap=ap)

        # Get epoch timings
        time_elapsed, time_per_epoch, time_left = utils.get_training_timings(
            self.start_time, self.current_epoch, self.args.max_epochs
        )
        # Print validation summary
        self.print_(f"Time for validation: {time() - val_start:.2f}s")
        self.print_(f"Epoch [{self.current_epoch}|{self.args.max_epochs}]"
                    f"\nTime elapsed: {time_elapsed}, "
                    f"Time left: {time_left}, "
                    f"Time per epoch: {time_per_epoch}"
                    f"\nVal - loss: {out['loss'].mean():.4f}, "
                    f"average precision: {ap:.4f}")

    def test_step(self, batch, batch_idx):
        x, y = batch

        # Save filename for evaluation
        name = x[0]
        anomaly = name.split('/')[-1].split(".nii.gz")[0][6:]

        x = x[1].unsqueeze(1)  # x was a tuple of (file_name, volume [slices, w, h])
        y = y[1].unsqueeze(1)  # y was a tuple of (file_name, volume [slices, w, h])
        feats, feats_rec = self(x)

        # Compute loss
        loss = self.loss_fn(feats_rec, feats).mean()

        # Get an anomaly map
        anomaly_map = self.compute_anomaly_map(feats_rec, feats)

        return {
            "inp":  x.cpu(),
            "anomaly": anomaly,
            "target_seg": y.cpu(),
            "loss": loss,
            "anomaly_map": anomaly_map.cpu(),
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
        pred_label = evaluation.fpi_sample_score_list(
            predictions=pred.reshape(-1, *self.args.volume_shape)
        )

        # Perform evaluation for all anomalies separately
        print("----- SAMPLE-WISE EVALUATION -----")
        evaluation.full_evaluation_sample(pred_label, target_label, anomalies)
        print("\n----- PIXEL-WISE EVALUATION -----")
        th = evaluation.full_evaluation_pixel(pred, target_seg, anomalies)

        if self.logger:
            # Log to tensorboard
            tb = self.logger.experiment

            # Binarize map
            th = 0.0772  # TODO: remove
            bin_map = torch.where(pred > th, 1., 0.)

            unique_anomalies = set(anomalies)
            unique_anomalies.discard("normal")
            for anomaly in unique_anomalies:
                print(f"Writing test sample images of {anomaly} to tensorboard")
                # Filter only relevant anomalies
                x = torch.cat([m for m, a in zip(inp, anomalies) if a == anomaly])
                p = torch.cat([m for m, a in zip(pred, anomalies) if a == anomaly])
                b = torch.cat([m for m, a in zip(bin_map, anomalies) if a == anomaly])
                t = torch.cat([m for m, a in zip(target_seg, anomalies) if a == anomaly])

                # Shuffle before plotting
                perm= torch.randperm(len(x))
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
                tb.add_figure(f"Test samples {anomaly}", fig, global_step=self.global_step)

            # Log precision-recall curve to tensorboard
            fig = evaluation.plot_prc(
                predictions=out["anomaly_map"],
                targets=target_seg
            )
            tb.add_figure("Precision-Recall Curve", fig, global_step=self.global_step)


def train(args, trainer, train_files):
    # Init lighning model
    if args.model_ckpt:
        utils.printer(f"Restoring checkpoint from {args.model_ckpt}", args.verbose)
        model = LitModel.load_from_checkpoint(args.model_ckpt, args=args)
    else:
        model = LitModel(args)

    # Tkinter backend fails on server, switching to Agg
    if not args.debug:
        matplotlib.use('Agg')

    # Load data
    utils.printer("Loading training data", args.verbose)
    t_start = time()

    # Split into train- and val-files
    random.shuffle(train_files)
    split_idx = int((1 - args.val_fraction) * len(train_files))
    training_files = train_files[:split_idx]
    val_files = train_files[split_idx:]

    # Create Datasets and Dataloaders
    train_ds = TrainDataset(training_files, args.img_size, args.slices_lower_upper)
    trainloader = DataLoader(train_ds, batch_size=args.batch_size,
                             num_workers=args.num_workers)
    val_ds = PatchSwapDataset(val_files, args.img_size, args.slices_lower_upper,
                              data=args.data)
    valloader = DataLoader(val_ds, batch_size=args.batch_size,
                           num_workers=args.num_workers, shuffle=True)

    utils.printer(f"Finished loading training data in {time() - t_start:.2f}s", args.verbose)

    # Train
    model.start_time = time()
    utils.printer("Start training", args.verbose)
    trainer.fit(model, trainloader, valloader)

    # Return the trained model
    return model


def test(args, trainer, test_files, model=None):
    # Init lighning model
    if args.model_ckpt is None and model is None:
        print("Warning: testing untrained model")
        model = LitModel(args)
    elif model is None:
        print(f"Restoring checkpoint from {args.model_ckpt}")
        model = LitModel.load_from_checkpoint(args.model_ckpt, args=args)

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
    parser.add_argument("--loss_fn", type=str, default='SSIM',
                        choices=["L2", "L1", "SSIM"])
    parser.add_argument("--no_train", dest="train", action="store_false")
    parser.add_argument("--no_test", dest="test", action="store_false")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--model_ckpt", type=str, default=None)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--val_every_epoch", type=int, default=1)
    parser.add_argument("--val_fraction", type=int, default=0.1)
    # Data params
    parser.add_argument("--data", type=str, default="brain",
                        choices=["brain", "abdom"])
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument('--slices_lower_upper',
                         nargs='+', type=int, default=[23, 200])
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
    parser.add_argument("--zdim", type=int, default=143)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--cnn_layers", type=str, nargs='+',
                        default=['layer1', 'layer2', 'layer3'])
    parser.add_argument("--keep_feature_prop", type=float, default=1.0)
    # Real Hyperparameters
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=256)
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

    # Save number of slices per sample as a parameters
    args.n_slices = args.slices_lower_upper[1] - args.slices_lower_upper[0]
    args.volume_shape = [args.n_slices, args.img_size, args.img_size]

    # Get train and test paths
    train_files = get_train_files(MOODROOT, args.data)
    test_files = get_test_files(MOODROOT, args.data)
    # train_files = train_files[:10]
    # test_files = test_files[:10]

    # Init logger
    if args.debug:
        logger = None
        callbacks = []
    else:
        logger = TensorBoardLogger(
            args.log_dir, name="fae", log_graph=True)
        # Add a ModelCheckpoint callback. Always log last ckpt and best train
        callbacks = [ModelCheckpoint(monitor=args.target_metric, mode='max',
                                     save_last=True)]

    # Init trainer
    trainer = pl.Trainer(gpus=args.gpus,
                         callbacks=callbacks,
                         logger=logger,
                         precision=args.precision,
                         # progress_bar_refresh_rate=0,  # Disable progress bar
                         checkpoint_callback=not args.debug,
                         check_val_every_n_epoch=args.val_every_epoch,
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
