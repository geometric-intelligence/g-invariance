import os

import pydantic
import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import pytorch_lightning.loggers as pl_loggers
import torch
import torch.nn.functional as F
import wandb
import yaml
from filelock import FileLock
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms as torchvision_transforms

import g_invariance.dataset as g_dataset
from g_invariance import model, rich_gi

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")


class Config(pydantic.BaseModel):
    class Config:
        extra = "allow"

    def __getitem__(self, item):
        return getattr(self, item)

    def update(self, new_config):
        for k, v in new_config.items():
            setattr(self, k, v)


def read_config_from_file(path: str = CONFIG_FILE) -> Config:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r") as _:
        obj = yaml.safe_load(_.read())
        return Config.parse_obj(obj)


class Classifier(pl.LightningModule):
    def __init__(self, config, wandb_setup_callback=None, extra_config=None):
        super(Classifier, self).__init__()
        self.accuracy = Accuracy(
            task="multiclass", num_classes=config.fc_sizes[-1], top_k=1
        )
        self.lr = config["lr"]
        self.model = getattr(model, config.model_name)(config)
        self.eval_loss = []
        self.eval_accuracy = []
        self.config = config
        self.param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.save_hyperparameters()
        self.max_accuracy = 0.0

    def on_fit_start(self):
        wandb.init(config=self.config)
        wandb.config.update({"params": self.param_count})

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        self.log("train_loss", loss, sync_dist=True)
        self.log("train_accuracy", accuracy, sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        self.eval_loss.append(loss)
        self.eval_accuracy.append(accuracy)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.eval_loss).mean()
        avg_acc = torch.stack(self.eval_accuracy).mean()
        self.log("ptl/val_loss", avg_loss, sync_dist=True)
        self.log("ptl/val_accuracy", avg_acc, sync_dist=True)
        self.log("param_count", self.param_count)
        # self.log("pooling_output_size", self.model.pooling_output_size)
        # self.log("first_layer_size", self.model.first_fc_size)
        if avg_acc > self.max_accuracy:
            self.max_accuracy = avg_acc
            self.log("ptl/max_val_accuracy", avg_acc, sync_dist=True)
        self.eval_loss.clear()
        self.eval_accuracy.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=1, T_mult=self.config.max_epochs, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "ptl/val_loss",
        }


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        if config.dataset_name in ["MNIST", "EMNIST", "FashionMNIST"]:
            normalize_transform = torchvision_transforms.Normalize((0.1307,), (0.3081,))
        elif config.dataset_name == "CIFAR10":
            normalize_transform = torchvision_transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
            )
        else:
            raise ValueError(f"Unknown dataset {config.dataset_name}")

        self.transforms = torchvision_transforms.Compose(
            [
                torchvision_transforms.ToTensor(),
                # TODO: Some issues when size changes. Understand why.
                torchvision_transforms.Resize((config.img_size,), antialias=True),
                normalize_transform,
            ]
        )
        self.num_workers = config.num_workers

    def setup(self, stage=None):
        if stage == "fit":
            with FileLock(f"{self.config.dataset_dir}/.lock"):
                # TODO: Train/Val datasets need splits with disjoint sets of angles.
                dataset = g_dataset.AugmentedDataset(
                    self.config.dataset_dir,
                    train=True,
                    group=self.config.data_augmentation,
                    transform=self.transforms,
                    n_samples=self.config.augmentation_factor,
                    dataset_name=self.config.dataset_name,
                    circle_crop=self.config.circle_crop,
                )
                val_count = int(len(dataset) * 0.2)
                train_count = len(dataset) - val_count
                self.data_train, self.data_val = random_split(
                    dataset, [train_count, val_count]
                )

                self.data_test = g_dataset.AugmentedDataset(
                    self.config.dataset_dir,
                    train=False,
                    group=self.config.data_augmentation,
                    transform=self.transforms,
                    n_samples=self.config.augmentation_factor,
                    dataset_name=self.config.dataset_name,
                    circle_crop=self.config.circle_crop,
                )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # Careful here, some incompatibilities have been noticed
            # with pin_memory=True
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
        )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    config = read_config_from_file()
    pl.seed_everything(config.seed)

    dm = MNISTDataModule(config)

    model = Classifier(config)

    learning_rate_monitor = pl_callbacks.LearningRateMonitor(logging_interval="step")

    trainer_callbacks = [
        learning_rate_monitor,
        rich_gi.progress_bar(),
    ]

    logger = pl_loggers.WandbLogger(log_model=False, project=config.name)

    trainer = pl.Trainer(
        devices=config.gpu_count,
        accelerator="gpu",
        strategy="ddp",
        enable_progress_bar=config.progress_bar,
        log_every_n_steps=config.log_every_n_steps,
        max_epochs=config.max_epochs,
        callbacks=trainer_callbacks,
        logger=logger,
    )
    input = torch.randn(1, config.img_size, config.img_size)
    trainer.fit(model, datamodule=dm)
