import copy
import hashlib
import inspect
import os
from collections import OrderedDict

import jsonpickle
import numpy as np
import pandas as pd
import torch
import wandb
from transform_datasets.dataset import TransformDataset

import gtc


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def gen_dataset(config):
    """
    Generate a TransformDataset from a config dictionary with the following
    structure:

    config = {
        "pattern": {"type": obj, "params": {}},
        "transforms": {
            "0": {"type": obj, "params": {}},
            "1": {"type": obj, "params": {}}
         }
    }

    The "type" parameter in each dictionary specifies an uninstantiated dataset
    or transform class. The "params" parameter specifies a dictionary containing
    the keyword arguments needed to instantiate the class.
    """
    if "seed" in config:
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
    # Catch for datasets and transforms that have no parameters
    if "params" not in config["pattern"]:
        config["pattern"]["params"] = {}
    for t in config["transforms"]:
        if "params" not in config["transforms"][t]:
            config["transforms"][t]["params"] = {}

    # Instantiate pattern object
    pattern = config["pattern"]["type"](**config["pattern"]["params"])

    # Instantiate transform objects
    transforms = [
        config["transforms"][k]["type"](**config["transforms"][k]["params"])
        for k in sorted(config["transforms"])
    ]

    # Generate dataset
    dataset = TransformDataset(pattern, transforms)
    return dataset


def get_names(config):
    dataset_type = config["pattern"]["type"].__name__
    dataset_hash = config_to_hash(config)

    transform_name = "-".join(
        [config["transforms"][k]["type"].__name__ for k in sorted(config["transforms"])]
    )
    dataset_name = dataset_type + "_" + transform_name
    return dataset_name, dataset_type, dataset_hash


def get_dataset_path(config, prefix):
    filename = hashlib.md5(jsonpickle.encode(config).encode("utf-8")).hexdigest()
    filename = f"{filename}.pt"
    return os.path.join(prefix, filename)


def create_dataset(config, prefix):
    if not os.path.exists(prefix):
        print(f"Creating directory {prefix}")
        os.makedirs(prefix)
    path = get_dataset_path(config, prefix)
    print(f"path={path}")
    if os.path.exists(path):
        print(f"Dataset already exists at {path}")
        return
    if "seed" in config:
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
    dataset = gen_dataset(config)
    torch.save(dataset, path)
    return


def load_dataset(config, prefix):
    path = get_dataset_path(config, prefix)
    dataset = torch.load(path)
    return dataset


class Config(dict):
    def __init__(self, config):
        """
        Takes in a dictionary config of the following form:

        config = {
            "type": Class,
            "params": {
                "param1": val,
                "param2": val
                }
        }
        """
        config = self.fill_defaults(config)
        super().__init__(**config)
        self.__dict__ = self

    def fill_defaults(self, config):
        defaults = get_default_args(config["type"])
        for k, v in defaults.items():
            if k not in config["params"]:
                config["params"][k] = v
        return config

    def build(self):
        return self["type"](**self["params"])


def load_checkpoint(logdir: str):
    checkpoint = torch.load(logdir)
    if not hasattr(checkpoint, "model"):
        trainer = checkpoint["trainer"]
        model_config = Config(trainer.logger.config["model"])
        optimizer_config = Config(copy.deepcopy(trainer.logger.config["optimizer"]))
        trainer.model = model_config.build()
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_config["params"]["params"] = trainer.model.parameters()
        trainer.optimizer = optimizer_config.build()
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        checkpoint = trainer
    return checkpoint


def create_model(block_configs, data_loader):
    blocks = OrderedDict()
    i = 0
    for name, config in block_configs.items():
        if i > 0:
            config.params["in_type"] = block.out_type  # Previous block
        if (
            config.type == gtc.modules.FullyConnectedBlock
            or config.type == gtc.modules.Linear
            or config.type == gtc.modules.BatchNorm1D
        ):
            x, y = next(iter(data_loader.train))
            with torch.no_grad():
                for k, b in blocks.items():
                    x = b(x)
                out_dim = x.shape[-1]
            config.params["in_dim"] = out_dim
        block = config.build()
        blocks[name] = block
        i += 1
    model = torch.nn.Sequential(blocks)
    return model


def load_checkpoint(logdir, device="cpu"):
    from gtc.utils import Config

    checkpoint = torch.load(logdir, map_location=device)
    trainer = checkpoint["trainer"]
    data_loader = Config(trainer.logger.config["data_loader"]).build()
    dataset_config = trainer.logger.config["dataset"]
    dataset = load_dataset(dataset_config)
    data_loader.load(dataset)
    if not hasattr(checkpoint, "model"):
        for k, v in trainer.logger.config["model"].items():
            trainer.logger.config["model"][k] = Config(v)
        model = create_model(trainer.logger.config["model"], data_loader)
        trainer.model = model
        trainer.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    optimizer_config = Config(copy.deepcopy(trainer.logger.config["optimizer"]))
    optimizer_config["params"]["params"] = trainer.model.parameters()
    trainer.optimizer = optimizer_config.build()
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint, trainer, data_loader


def load_wandb_checkpoint(entity, project, run_id, epoch=None):
    if epoch is None:
        api = wandb.Api()
        run = api.run("{}/{}/{}".format(entity, project, run_id))
        epoch = run.summary.epoch
        loaded = False
        while not loaded:
            if epoch < 0:
                raise Exception("No saved checkpoints.")
            try:
                checkpoint_path = wandb.restore(
                    "checkpoints/checkpoint_{}.pt".format(epoch),
                    run_path="{}/{}/{}".format(entity, project, run_id),
                ).name
                loaded = True
            except:
                epoch -= 1
    else:
        checkpoint_path = wandb.restore(
            "checkpoints/checkpoint_{}.pt".format(epoch),
            run_path="{}/{}/{}".format(entity, project, run_id),
        ).name
    checkpoint, trainer, data_loader = load_checkpoint(checkpoint_path)
    return checkpoint, trainer, data_loader


def nest_dict(dict1):
    result = {}
    for k, v in dict1.items():
        # for each key call method split_rec which
        # will split keys to form recursively
        # nested dictionary
        split_rec(k, v, result)
    return result


def split_rec(k, v, out, sep="."):
    # splitting keys in dict
    # calling_recursively to break items on '_'
    k, *rest = k.split(sep, 1)
    if rest:
        split_rec(rest[0], v, out.setdefault(k, {}))
    else:
        out[k] = v


def flatten_dict(dd, separator=".", prefix=""):
    return (
        {
            prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
        }
        if isinstance(dd, dict)
        else {prefix: dd}
    )


def config_to_hash(config):
    if type(config) != dict:
        config = dict(config)
    flat_config = pd.json_normalize(config).to_dict()
    flat_config = sorted(flat_config.items())
    config_hash = hashlib.md5(jsonpickle.encode(flat_config).encode("utf-8")).digest()
    return config_hash.hex()


def nested_set(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def nested_get(dic, keys):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    return dic[keys[-1]]


def fix_wandb_config(wandb_config, master_config):
    new_config = copy.deepcopy(master_config)
    for k, v in wandb_config.items():
        key_list = k.split(".")
        master_val = nested_get(new_config, key_list)
        if v != "random":
            try:
                exec("import {}".format(v.split(".")[0]))
                v = eval(v)
                imported = True
                nested_set(new_config, key_list, v)
            except:
                imported = False
        else:
            imported = False
        if not imported:
            if type(master_val) == type or callable(master_val):
                continue
            else:
                nested_set(new_config, key_list, v)
    return new_config


def run_trainer(  # previously device=0
    master_config,
    logger_config,
    device=torch.device("cuda"),
    n_examples=1e9,
    entity=None,
    project=None,
    prefix="dataset",
):
    flat_config = flatten_dict(master_config)
    with wandb.init(config=flat_config, entity=entity, project=project) as run:
        new_config = fix_wandb_config(wandb.config, master_config)
        # Nina.

        dataset = load_dataset(master_config["dataset"], prefix=prefix)

        data_loader = new_config["data_loader"].build()
        data_loader.load(dataset)
        trainer = construct_trainer(
            master_config, logger_config, new_config, data_loader
        )

        epochs = int(n_examples // len(data_loader.train.dataset.data))
        print(f"device: {device}")
        trainer.model.device = device
        trainer.model = trainer.model.to(device)
        num_params = sum(
            param.numel() for param in trainer.model.parameters() if param.requires_grad
        )
        print(f"Number of parameters: {num_params}")
        print(f"Number of epochs: {epochs}")

        trainer.train(data_loader, epochs=epochs)


def construct_trainer(master_config, logger_config, wandb_config, data_loader):
    """
    master_config has the following format:

    master_config = {
        "dataset": dataset_config,
        "model": model_config,
        "optimizer": optimizer_config,
        "loss": loss_config,
        "data_loader": data_loader_config,
    }

    with optional regularizer, normalizer, and learning rate scheduler
    """

    torch.manual_seed(wandb_config["seed"])
    np.random.seed(wandb_config["seed"])

    # CURRENTLY, SWEEPS ON MODEL HYPERPARAMS WILL NOT WORK
    model = create_model(master_config["model"], data_loader)

    loss = wandb_config["loss"].build()

    logger_config["params"]["config"] = wandb_config
    logger = logger_config.build()

    optimizer_config = copy.deepcopy(wandb_config["optimizer"])
    optimizer_config["params"]["params"] = model.parameters()
    optimizer = optimizer_config.build()

    if "trainer" not in master_config:
        trainer_type = Trainer
    else:
        trainer_type = master_config["trainer"]

    train_config = Config(
        {
            "type": trainer_type,
            "params": {
                "model": model,
                "loss": loss,
                "logger": logger,
                "optimizer": optimizer,
            },
        }
    )

    if "regularizer" in wandb_config:
        regularizer = wandb_config["regularizer"].build()
        train_config["params"]["regularizer"] = regularizer

    if "normalizer" in wandb_config:
        normalizer = wandb_config["normalizer"].build()
        train_config["params"]["normalizer"] = normalizer

    if "scheduler" in wandb_config:
        scheduler_config = copy.deepcopy(wandb_config["scheduler"])
        scheduler_config["params"]["optimizer"] = optimizer
        scheduler = scheduler_config.build()
        train_config["params"]["scheduler"] = scheduler

    trainer = train_config.build()

    return trainer


class WBLogger:
    def __init__(
        self,
        config,
        project=None,
        data_project=None,
        entity=None,
        watch_interval=1,
        log_interval=1,
        plot_interval=1,
        checkpoint_interval=10,
        step_plotter=None,
        end_plotter=None,
    ):
        """
        watch_interval is in number of batches
        log_interval is in number of epochs
        """
        self.project = project
        self.data_project = data_project
        self.entity = entity
        self.config = config
        self.watch_interval = watch_interval
        self.log_interval = log_interval
        self.plot_interval = plot_interval
        self.checkpoint_interval = checkpoint_interval
        self.step_plotter = step_plotter
        self.end_plotter = end_plotter
        self.is_finished = False

    def begin(self, model, data_loader):
        wandb.watch(model, log_freq=self.watch_interval, log_graph=False)
        os.makedirs(os.path.join(wandb.run.dir, "checkpoints"), exist_ok=True)

    def log_step(
        self,
        trainer,
        log_dict,
        variable_dict,
        epoch,
        val_log_dict=None,
        n_examples=None,
    ):
        full_log_dict = {}
        if epoch % self.log_interval == 0:
            if val_log_dict is not None:
                for k in log_dict:
                    full_log_dict["train_" + k] = log_dict[k]
                    full_log_dict["val_" + k] = val_log_dict[k]
            else:
                full_log_dict = log_dict

            full_log_dict["epoch"] = epoch
            if n_examples is not None:
                full_log_dict["n_examples"] = n_examples

            if (
                self.step_plotter is not None
                and variable_dict is not None
                and (epoch % self.plot_interval == 0)
            ):
                plots = self.step_plotter.plot(variable_dict)
                full_log_dict.update(plots)

            wandb.log(full_log_dict, step=epoch)
        if epoch % self.checkpoint_interval == 0:
            self.save_checkpoint(trainer, epoch)

    def end(self, trainer, variable_dict, epoch):
        if self.end_plotter is not None:
            plots = self.end_plotter.plot(variable_dict)
            wandb.log(plots, step=epoch)

        self.save_checkpoint(trainer, epoch)

        wandb.finish()
        self.is_finished = True

    def save_checkpoint(self, trainer, iter):
        checkpoint = {
            "trainer": trainer,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
        }
        torch.save(
            checkpoint,
            os.path.join(wandb.run.dir, "checkpoints", "checkpoint_{}.pt".format(iter)),
        )
        wandb.save(
            os.path.join(wandb.run.dir, "checkpoints", "checkpoint_{}.pt".format(iter)),
            base_path=wandb.run.dir,
            policy="now",
        )


class TrainValLoader:
    def __init__(self, batch_size, fraction_val=0.2, num_workers=0, seed=0):
        assert (
            fraction_val <= 1.0 and fraction_val >= 0.0
        ), "fraction_val must be a fraction between 0 and 1"

        np.random.seed(seed)

        self.batch_size = batch_size
        self.fraction_val = fraction_val
        self.seed = seed
        self.num_workers = num_workers

    def split_data(self, dataset):

        if self.fraction_val > 0.0:
            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(self.fraction_val * len(dataset)))

            np.random.shuffle(indices)

            train_indices, val_indices = indices[split:], indices[:split]
            val_dataset = copy.deepcopy(dataset)
            val_dataset.data = val_dataset.data[val_indices]
            val_dataset.labels = val_dataset.labels[val_indices]

            train_dataset = copy.deepcopy(dataset)
            train_dataset.data = train_dataset.data[train_indices]
            train_dataset.labels = train_dataset.labels[train_indices]

        else:
            val_dataset = None

        return train_dataset, val_dataset

    def construct_data_loaders(self, train_dataset, val_dataset):
        if val_dataset is not None:
            val = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )

        else:
            val = None

        train = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return train, val

    def load(self, dataset):
        train_dataset, val_dataset = self.split_data(dataset)
        self.train, self.val = self.construct_data_loaders(train_dataset, val_dataset)


class Trainer(torch.nn.Module):
    def __init__(
        self,
        model,
        loss,
        optimizer,
        logger=None,
        scheduler=None,
        regularizer=None,
        normalizer=None,
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.logger = logger
        self.regularizer = regularizer
        self.normalizer = normalizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = 0
        self.n_examples = 0

    def step(self, data_loader, grad=True):
        """Compute a single step of training.

        This example is a minimal implementation of the `step` function
        for a classification problem with a simple regularization on
        the model parameters.

        Parameters
        ----------
        data_loader : torch.utils.data.dataloader.DataLoader
        grad : boolean
            required argument to switch between training and evaluation

        Returns
        -------
        log_dict : dictionary with losses to be logged by the trainer/logger
            format - {'total_loss': total_loss, 'l1_penalty': l1_penalty, ...}
            Your dictionary must contain a key called `total_loss`

        """
        log_dict = {"loss": 0, "reg_loss": 0, "total_loss": 0}
        for i, (x, labels) in enumerate(data_loader):
            loss = 0
            reg_loss = 0
            total_loss = 0

            x = x.to(self.model.device)
            labels = labels.to(self.model.device)

            if grad:
                self.optimizer.zero_grad()
                out = self.model.forward(x)

            else:
                with torch.no_grad():
                    out = self.model.forward(x)

            # Compute loss term without regularization terms (e.g. classification loss)
            loss += self.loss(out, labels)
            log_dict["loss"] += loss
            total_loss += loss

            # Compute regularization penalty terms (e.g. sparsity, l2 norm, etc.)
            if self.regularizer:
                reg_variable_dict = {
                    "x": x,
                    "out": out,
                } | dict(
                    self.model.named_parameters()
                )  # Must use named parameters rather than state_dict to preserve grads

                reg_loss += self.regularizer(reg_variable_dict)
                log_dict["reg_loss"] += reg_loss
                total_loss += reg_loss

            if grad:
                total_loss.backward()
                self.optimizer.step()

            if self.normalizer is not None:
                self.normalizer(dict(self.model.named_parameters()))

            log_dict["total_loss"] += total_loss

        # Normalize loss terms for the number of samples/batches in the epoch (optional)
        n_samples = len(data_loader)
        for key in log_dict.keys():
            log_dict[key] /= n_samples

        plot_variable_dict = {"model": self.model}

        return log_dict, plot_variable_dict

    def train(
        self,
        data_loader,
        epochs,
        start_epoch=0,
        print_status_updates=True,
        print_interval=1,
    ):
        if self.logger is not None:
            self.logger.begin(self.model, data_loader)

        try:
            for i in range(start_epoch, start_epoch + epochs + 1):
                self.epoch = i
                log_dict, plot_variable_dict = self.step(data_loader.train, grad=True)

                if data_loader.val is not None:
                    # By default, plots are only generated on train steps
                    val_log_dict, _ = self.evaluate(data_loader.val)
                else:
                    val_log_dict = None

                if self.scheduler is not None:
                    if val_log_dict is not None:
                        self.scheduler.step(val_log_dict["total_loss"])
                    else:
                        self.scheduler.step(train_log_dict["total_loss"])

                if self.logger is not None:
                    self.logger.log_step(
                        trainer=self,
                        log_dict=log_dict,
                        val_log_dict=val_log_dict,
                        variable_dict=plot_variable_dict,
                        epoch=self.epoch,
                        n_examples=self.n_examples,
                    )

                if i % print_interval == 0 and print_status_updates == True:
                    if data_loader.val is not None:
                        self.print_update(log_dict, val_log_dict)
                    else:
                        self.print_update(log_dict)

                self.n_examples += len(data_loader.train.dataset)

        except KeyboardInterrupt:
            print("Stopping and saving run at epoch {}".format(i))
        end_dict = {"model": self.model, "data_loader": data_loader}
        if self.logger is not None:
            self.logger.end(self, end_dict, self.epoch)

    def resume(self, data_loader, epochs):
        self.train(data_loader, epochs, start_epoch=self.epoch + 1)

    @torch.no_grad()
    def evaluate(self, data_loader):
        results = self.step(data_loader, grad=False)
        return results

    def print_update(self, result_dict_train, result_dict_val=None):

        update_string = "Epoch {} ||  N Examples {} || Train Total Loss {:0.5f}".format(
            self.epoch, self.n_examples, result_dict_train["total_loss"]
        )
        if result_dict_val:
            update_string += " || Validation Total Loss {:0.5f}".format(
                result_dict_val["total_loss"]
            )
        print(update_string)
