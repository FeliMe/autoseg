import argparse
import os
from time import time
from warnings import warn

import matplotlib
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from ray import tune
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
import wandb

from uas_mood.models import models
from uas_mood.utils import evaluation, utils
from uas_mood.utils.dataset import CXR14PatchSwapDataset, CXR14ROOT, CXR14TestDataset
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


class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        # Save all input args to self.hparams
        self.save_hyperparameters()
        self.args = self.hparams.args

        # Network
        self.net = models.VAE(img_size=torch.tensor([args.img_size, args.img_size]),
                              model_width=args.model_width)
        self.net.apply(models.weights_init_relu)

        # Example input array needed to log the graph in tensorboard
        input_size = (1, self.args.img_size, self.args.img_size)
        self.example_input_array = torch.randn(
            [5, *input_size])

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
        #         'scheduler': torch.optim.lr_scheduler.StepLR(
        #             optimizer=optimizer,
        #             step_size=5,
        #             gamma=0.1
        #         ),
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

    def plot_reconstruction(self, inp, anomaly_map, num_images=-1):
        images = [
            make_grid(inp.cpu()[:num_images], padding=0),
            make_grid(anomaly_map.cpu()[:num_images], padding=0),
        ]
        titles = [
            "Input",
            "Anomaly map",
        ]

        # Log
        self.logger.experiment.log({
            "val samples": [
                wandb.Image(im, caption=t) for im, t in zip(images, titles)
            ]
        })

    def training_step(self, batch, batch_idx):
        inp, _ = batch
        rec, mu, log_var = self(inp)

        # Compute loss
        loss, rec_loss, kl_loss = self.net.loss_function(rec, inp, mu, log_var,
                                                         self.args.M_N)

        self.log("train/loss", loss.cpu(), on_step=True)
        self.log("train/rec_loss", rec_loss.cpu(), on_step=True)
        self.log("train/kl_loss", kl_loss.cpu(), on_step=True)

        return {"loss": loss.cpu(),
                "rec_loss": rec_loss.detach().cpu(),
                "kl_loss": kl_loss.detach().cpu()}

    def training_epoch_end(self, outputs):
        # Stack all values to a dict
        out = self.stack_outputs(outputs)

        mean_loss = out["loss"].mean()
        mean_rec_loss = out["rec_loss"].mean()
        mean_kl_loss = out["kl_loss"].mean()

        # Print training epoch summary
        self.print_(
            f"Epoch [{self.current_epoch}/{self.args.max_epochs}]" \
            f" Train loss: {mean_loss:.4f}" \
            f" train rec loss: {mean_rec_loss:.4f}" \
            f" train kl loss: {mean_kl_loss:.4f}")

        # Tensorboard logs
        self.log_metric("train/epoch_loss", mean_loss)
        self.log_metric("train/epoch_rec_loss", mean_rec_loss)
        self.log_metric("train/epoch_kl_loss", mean_kl_loss)

    def validation_step(self, batch, batch_idx):
        inp, y = batch
        rec, mu, log_var = self(inp)

        # Compute loss
        loss, rec_loss, kl_loss = self.net.loss_function(rec, inp, mu, log_var,
                                                         self.args.M_N)

        anomaly_map = (inp - rec).abs()
        anomaly_score = anomaly_map.sum((1, 2, 3))

        return {
            "inp": inp.cpu(),
            "label": y.cpu(),
            "loss": loss.cpu(),
            "rec_loss": rec_loss.detach().cpu(),
            "kl_loss": kl_loss.detach().cpu(),
            "anomaly_map": anomaly_map.cpu(),
            "anomaly_score": anomaly_score.cpu(),
        }

    def validation_epoch_end(self, outputs):
        self.print_("Validating")
        val_start = time()
        # Stack all values to a dict
        out = self.stack_outputs(outputs)

        # Compute average precision
        ap = evaluation.compute_average_precision(
            out["anomaly_score"], out["label"])

        self.log_metric("val ap", ap)
        self.log_metric("val_loss", out["loss"].mean())

        # Tensorboard logs
        if self.logger:
            # log a reconstructed sample
            self.print_("Logging a validation sample to w&b")
            self.plot_reconstruction(
                out["inp"], out["anomaly_map"], num_images=self.args.num_images_log)

        # Log to tune if hyperparameter search
        if self.args.hparam_search:
            tune.report(ap=ap)

        # Get epoch timings
        time_elapsed, time_per_epoch, time_left = utils.get_training_timings(
            self.start_time, self.current_epoch * args.val_every_epoch, self.args.max_epochs
        )
        # Print validation summary
        self.print_(f"Val loss: {out['loss'].mean():.4f}, "
                    f"average precision val: {ap:.4f}\n"
                    f"Time elapsed: {time_elapsed}, "
                    f"Time left: {time_left}, "
                    f"Time per epoch: {time_per_epoch}, "
                    f"Time for validation: {time() - val_start:.2f}s")

    def test_step(self, batch, batch_idx):
        inp, y = batch
        rec = self(inp)[0]
        anomaly_map = (inp - rec).abs()
        anomaly_score = anomaly_map.sum((1, 2, 3))

        return {
            "inp": inp.cpu(),
            "label": y.cpu(),
            "anomaly_map": anomaly_map.cpu(),
            "anomaly_score": anomaly_score.cpu(),
        }

    def test_epoch_end(self, outputs):
        # Stack all values to a dict
        out = self.stack_outputs(outputs)

        # Unpack outputs
        inp = out["inp"]
        label = out["label"]
        anomaly_score = out["anomaly_score"]
        anomaly_map = out["anomaly_map"]

        # Perform evaluation for all anomalies separately
        print("----- SAMPLE-WISE EVALUATION -----")
        auroc, ap = evaluation.evaluate_sample_wise(anomaly_score, label)

        if self.logger:
            # Save summary metrics
            wandb.run.summary["test ap"] = ap
            wandb.run.summary["test auroc"] = auroc

            # Write samples to w&b
            print("Writing test sample images to w&b")

            # Filter positives
            is_positive = torch.where(label == 1)
            pos_inp = inp[is_positive]
            pos_anomaly_map = anomaly_map[is_positive]
            pos_anomaly_score = anomaly_score[is_positive]

            # Write best pos images
            best_idx = torch.sort(pos_anomaly_score)[1][-10:]
            best_pos_inp = pos_inp[best_idx]
            best_pos_anomaly_map = pos_anomaly_map[best_idx]

            best_pos_images = [best_pos_inp, best_pos_anomaly_map]
            best_pos_titles = ["Best positive input image",
                               "Best positive anomaly map"]

            # Log to w&b
            self.logger.experiment.log({
                "test_samples": [
                    wandb.Image(im, caption=t) for im, t in zip(best_pos_images, best_pos_titles)
                ]
            })

            # Write worst pos images
            worst_idx = torch.sort(pos_anomaly_score)[1][:10]
            worst_pos_inp = pos_inp[worst_idx]
            worst_pos_anomaly_map = pos_anomaly_map[worst_idx]

            worst_pos_images = [worst_pos_inp, worst_pos_anomaly_map]
            worst_pos_titles = ["Worst positive input image",
                                "Worst positive anomaly map"]

            # Log to w&b
            self.logger.experiment.log({
                "test_samples": [
                    wandb.Image(im, caption=t) for im, t in zip(worst_pos_images, worst_pos_titles)
                ]
            })


def train(args, trainer, train_files, test_files_normal, test_files_anomal):
    # # Split into train- and val-files
    # random.shuffle(train_files)
    # split_idx = int((1 - args.val_fraction) * len(train_files))
    # training_files = train_files[:split_idx]
    # val_files = train_files[split_idx:]
    # utils.printer(f"Training with {len(training_files)} files", args.verbose)
    # utils.printer(f"Validating on {len(val_files)} files", args.verbose)

    # # Create Datasets and Dataloaders
    # train_ds = CXR14PatchSwapDataset(training_files, args.img_size)
    # trainloader = DataLoader(train_ds, batch_size=args.batch_size,
    #                          num_workers=0, shuffle=True)
    # val_ds = CXR14PatchSwapDataset(val_files, args.img_size)
    # valloader = DataLoader(val_ds, batch_size=args.batch_size,
    #                        num_workers=0, shuffle=True)

    # Create validation set
    val_files_normal = test_files_normal[:int(args.val_fraction * len(test_files_normal))]
    val_files_anomal = test_files_anomal[:int(args.val_fraction * len(test_files_anomal))]
    val_files = val_files_normal + val_files_anomal
    val_labels = ([0] * len(val_files_normal)) + ([1] * len(val_files_anomal))

    train_ds = CXR14PatchSwapDataset(train_files, args.img_size,
                                     anomaly_shape=args.anomaly_shape,
                                     num_anomalies=0)
    trainloader = DataLoader(train_ds, batch_size=args.batch_size,
                             num_workers=0, shuffle=True)
    val_ds = CXR14TestDataset(val_files, val_labels, args.img_size)
    valloader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    args.M_N = args.batch_size / len(train_ds)

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
        # Make wandb_logger watch model
        trainer.logger.watch(model)

    # Train
    model.start_time = time()
    utils.printer("Start training", args.verbose)
    trainer.fit(model, trainloader, valloader)

    # Return the trained model
    return model


def test(args, trainer, test_files_normal, test_files_anomal, model=None):
    # Init lighning model
    if args.model_ckpt is None and model is None:
        warn("Evaluating untrained model")
        model = LitModel(args)
    elif model is None:
        print(f"Restoring checkpoint from {args.model_ckpt}")

        # download checkpoint locally (if not already cached)
        run = wandb.init(project="mood_cxr14")
        args.model_ckpt = f"felix-meissen/mood_cxr14/model-{args.model_ckpt}:v0"
        artifact = run.use_artifact(args.model_ckpt, type="model")
        artifact_dir = artifact.download()
        ckpt = os.path.join(artifact_dir, "model.ckpt")
        # load checkpoint
        model = LitModel.load_from_checkpoint(ckpt, args=args)

    # Load data
    print("Loading data")
    test_labels_normal = [0 for _ in range(len(test_files_normal))]
    test_labels_anomal = [1 for _ in range(len(test_files_anomal))]
    test_files = test_files_normal + test_files_anomal
    test_labels = test_labels_normal + test_labels_anomal
    ds = CXR14TestDataset(test_files, test_labels, args.img_size)
    dl = DataLoader(ds, batch_size=128, num_workers=8)

    # Test
    print("Testing model")
    trainer.test(model, dl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # General script control params
    parser.add_argument("--no_train", dest="train", action="store_false")
    parser.add_argument("--no_test", dest="test", action="store_false")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--model_ckpt", type=str, default=None,
                        help="felix-meissen/mood_cxr/model-...:v0")
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--val_every_epoch", type=float, default=1)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--no_load_to_ram", dest="load_to_ram",
                        action="store_false")
    # Model params
    parser.add_argument("--model", type=str, default="VAE")
    parser.add_argument("--model_width", type=int, default=16)
    # Data params
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--gender", type=str, default="Female",
                        choices=["Female", "Male"])
    parser.add_argument("--anomaly_shape", type=str, default="polygon",
                        choices=["rectangle", "polygon", "ellipse"])
    # Engineering params
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--precision", type=int, default=32)
    # Logging params
    parser.add_argument("--log_dir", type=str,
                        default=f"{os.path.dirname(os.path.abspath(__file__))}/logs/cxr14")
    parser.add_argument("--num_images_log", type=int, default=8)
    parser.add_argument("--run_name", "-r", type=str, default=None)
    # Ray tune params
    parser.add_argument("--hparam_search", action="store_true")
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--cpu_per_trial", type=int, default=12)
    parser.add_argument("--gpu_per_trial", type=float, default=0.5)
    parser.add_argument("--target_metric", type=str, default="ap")
    # Real Hyperparameters
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
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
        check_val_every_n_epoch = int(args.val_every_epoch)
        val_check_interval = 1.0

    # Get train and test paths
    train_files = utils.read_list_file_to_abs_path(
        os.path.join(CXR14ROOT, f"train_lists/norm_{args.gender}AdultPA_train_list.txt"),
        base=os.path.join(CXR14ROOT, "images")
    )
    test_files_normal = utils.read_list_file_to_abs_path(
        os.path.join(CXR14ROOT, f"test_lists/norm_{args.gender}AdultPA_test_list.txt"),
        base=os.path.join(CXR14ROOT, "images")
    )
    test_files_anomal = utils.read_list_file_to_abs_path(
        os.path.join(CXR14ROOT, f"test_lists/anomaly_{args.gender}AdultPA_test_list.txt"),
        base=os.path.join(CXR14ROOT, "images")
    )

    if args.max_train_files is not None:
        train_files = train_files[:args.max_train_files]

    if args.debug:
        train_files = train_files[:400]
        # test_files = test_files[:10]

    callbacks = [LitProgressBar()]

    # Init logger
    if args.debug:
        logger = None
    else:
        logger = WandbLogger(project="mood_cxr14", save_dir=args.log_dir,
                             log_model=True)
        # Add a ModelCheckpoint callback. Always log last ckpt and best train
        # callbacks += [ModelCheckpoint(monitor=args.target_metric, mode='max',
        #                               save_last=True, every_n_epochs=4)]

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
                         max_epochs=args.max_epochs)

    if args.hparam_search:
        # Hyperparameter search with ray tune
        search_config = {
            "lr": tune.loguniform(1e-5, 1e-3),
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
            model = train(args, trainer, train_files, test_files_normal, test_files_anomal)
        if args.test:
            test(args, trainer, test_files_normal, test_files_anomal, model=model)
