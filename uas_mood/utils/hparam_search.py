from argparse import Namespace
from functools import partial

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


def update_args(args: Namespace, config: dict) -> Namespace:
    args = vars(args)
    for key, value in config.items():
        args[key] = value
    return Namespace(**args)


def hparam_search_run(config, args, trainer, train_files, train_fn):
    # Add the config to args
    args = update_args(args, config)

    # Set model training to silent
    args.verbose = False

    # Ray tune doesn't need to do the full max_epochs
    # trainer.min_epochs = 1

    # Call train function
    train_fn(args, trainer, train_files)


def hparam_search(search_config, args, trainer, train_files, train_fn):
    # Hyperparameter search with ray tune
    scheduler = ASHAScheduler(
        metric=args.target_metric,
        mode="max",
        max_t=args.max_epochs,
        grace_period=6,  # Run experiments at least this epochs
        reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=list(search_config.keys()),
        metric_columns=[args.target_metric, "training_iteration"])
    result = tune.run(
        partial(hparam_search_run, trainer=trainer, train_files=train_files,
                args=args, train_fn=train_fn),
        resources_per_trial={
            "cpu": args.cpu_per_trial,
            "gpu": args.gpu_per_trial
        },
        config=search_config,
        num_samples=args.num_trials,  # Number of experiments
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial(args.target_metric, "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final auroc: {}".format(
        best_trial.last_result[args.target_metric]))
