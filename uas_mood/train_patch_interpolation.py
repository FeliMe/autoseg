import argparse
import os
from time import time

import matplotlib
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler

from uas_mood.models.models import WideResNetAE
from uas_mood.utils import evaluation
from uas_mood.utils import utils
from uas_mood.utils.dataset import (
    MOODROOT,
    PatchSwapDataset,
    TestDataset,
    get_test_files,
    get_train_files,
)


class LitModel(pl.LightningModule):
    def __init__(self, args=None):
        super().__init__()

        # Save all args
        self.args = args

        # Network
        self.net = WideResNetAE(inp_size=args.img_size, widen_factor=args.model_width)

        # Example input array needed to log the graph in tensorboard
        self.example_input_array = torch.randn(
            [5, 1, args.img_size, args.img_size])

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

    @staticmethod
    def stack_outputs(outputs):
        """concatenates a list of of dictionaries with torch.tensors to a
        single dict"""
        out = {}
        for key in outputs[0].keys():
            if outputs[0][key].ndim == 0:
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
        loss = F.binary_cross_entropy(pred, y)

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        # Stack all values to a dict
        out = self.stack_outputs(outputs)

        # Print training epoch summary
        self.print_(f"Epoch [{self.current_epoch}/{self.args.max_epochs}] Train - loss: {out['loss'].mean():.4f}")

        # Tensorboard logs
        if self.logger:
            self.log("train_loss", out["loss"].mean(), logger=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        # Compute loss
        loss = F.binary_cross_entropy(pred, y)

        return {
            "inp":  x.cpu(),
            "target_seg": y.cpu(),
            "loss": loss,
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

        # Tensorboard logs
        if self.logger:
            self.log("ap", ap, logger=True)
            self.log("val_loss", out['loss'].mean(), logger=True)

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
        print(f"Time for validation: {time() - val_start:.2f}s")
        self.print_(f"Epoch [{self.current_epoch}|{self.args.max_epochs}]"
                    f"\nTime elapsed: {time_elapsed}, "
                    f"Time left: {time_left}, "
                    f"Time per epoch: {time_per_epoch}"
                    f"\nVal - loss: {out['loss'].mean():.4f}, "
                    f"average precision: {ap:.4f}")

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        # Compute loss
        loss = F.binary_cross_entropy(pred, y)

        return {
            "inp":  x.cpu(),
            "target_seg": y.cpu(),
            "loss": loss,
            "anomaly_map": pred.cpu(),
        }

    def test_epoch_end(self, outputs):
        # Stack all values to a dict
        out = self.stack_outputs(outputs)

        target_seg = out['target_seg']

        _, _, th = evaluation.evaluate(
            predictions=out['anomaly_map'],
            targets=target_seg,
        )

        if self.logger:
            """Log to tensorboard"""
            tb = self.logger.experiment

            # Get binary label per image
            label = torch.where(target_seg.sum((1, 2, 3)) > 0, 1, 0)

            # Binarize map
            bin_map = torch.where(out['anomaly_map'] > th, 1., 0.)

            images = [
                out['inp'][label == 1],
                out['anomaly_map'][label == 1],
                bin_map[label == 1],
            ]
            titles = [
                'Input image',
                'Anomaly map',
                'Binarized map',
            ]

            if out['label'].ndim > 1:
                images.append(out['label'][label == 1])
                titles.append('Ground truth')

            # Log sample images to tensorboard
            print("Writing test sample images to tensorboard")
            fig = evaluation.plot_results(
                images=images,
                titles=titles,
                n_images=10
            )
            tb.add_figure("Test samples", fig, global_step=self.global_step)

            # Log precision-recall curve to tensorboard  # TODO: Implement
            # fig = evaluation.plot_roc(fpr, tpr, auroc, title="Precision-Recall Curve")
            # tb.add_figure("Precision-Recall Curve", fig, global_step=self.global_step)


def train(args, trainer, train_files):
    # Load data
    print("Loading training data")
    t_start = time()
    ds = PatchSwapDataset(train_files, args.img_size, args.slices_lower_upper)
    trainloader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers)
    val_inds = torch.randperm(len(ds))[:args.val_samples]
    val_sampler = SubsetRandomSampler(val_inds)
    valloader = DataLoader(ds, batch_size=args.batch_size,
                           num_workers=args.num_workers, sampler=val_sampler)
    print(f"Finished loading training data in {time() - t_start:.2f}s")

    # Init lighning model
    if args.model_ckpt:
        print(f"Restoring checkpoint from {args.model_ckpt}")
        model = LitModel.load_from_checkpoint(args.model_ckpt, args=args)
    else:
        model = LitModel(args)

    # Tkinter backend fails on server, switching to Agg
    if not args.debug:
        matplotlib.use('Agg')

    # Train
    model.start_time = time()
    print("Start training")
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
    parser.add_argument("--no_train", dest="train", action="store_false")
    parser.add_argument("--no_test", dest="test", action="store_false")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--model_ckpt", type=str, default=None)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--val_every_epoch", type=int, default=5)
    parser.add_argument("--val_samples", type=int, default=64 * 5)
    # Data params
    parser.add_argument("--data", type=str, default="brain",
                        choices=["brain", "abdom"])
    parser.add_argument("--img_size", type=int, default=128)
    # parser.add_argument('--slices_lower_upper',
    #                      nargs='+', type=int, default=[23, 200])
    parser.add_argument('--slices_lower_upper',
                        nargs='+', type=int, default=[79, 83])
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
    # Real Hyperparameters
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_width", type=int, default=4)
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

    # Get train and test paths
    train_files = get_train_files(MOODROOT, args.data)
    test_files = get_test_files(MOODROOT, args.data)

    # Init logger
    if args.debug:
        logger = None
        callbacks = []
    else:
        logger = TensorBoardLogger(
            args.log_dir, name="patch_interpolation", log_graph=True)
        # Add a ModelCheckpoint callback. Always log last ckpt and best train
        callbacks = [ModelCheckpoint(monitor=args.target_metric, mode='max',
                                     save_last=True)]
        # callbacks = []

    # Init trainer
    trainer = pl.Trainer(gpus=args.gpus,
                         callbacks=callbacks,
                         logger=logger,
                         precision=args.precision,
                         progress_bar_refresh_rate=0,  # Disable progress bar
                         checkpoint_callback=not args.debug,
                         check_val_every_n_epoch=args.val_every_epoch,
                         num_sanity_val_steps=0,
                         # min_epochs=args.max_epochs,
                         max_epochs=args.max_epochs)

    if args.hparam_search:
        # Hyperparameter search with ray tune
        raise NotImplementedError
        # hparam_search(
        #     args=args,
        #     trainer=trainer,
        #     train_fn=train
        # )
    else:
        model = None
        if args.train:
            model = train(args, trainer, train_files)
        if args.test:
            test(args, trainer, test_files, model=model)
