import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import pytorch_lightning.loggers as pl_loggers
import torch
import torch.nn.functional as F
from filelock import FileLock
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision import datasets


import os

import pydantic
import yaml

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")


class Config(pydantic.BaseModel):

    class Config:
        extra = "allow"

    def __getitem__(self, item):
        return getattr(self, item)


def read_config_from_file(path: str = CONFIG_FILE) -> Config:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r") as _:
        obj = yaml.safe_load(_.read())
        return Config.parse_obj(obj)


class MNISTClassifier(pl.LightningModule):
    def __init__(self, config):
        super(MNISTClassifier, self).__init__()
        self.accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1)
        self.lr = config["lr"]
        self.model = model.Net(config)
        self.eval_loss = []
        self.eval_accuracy = []
        self.config = config

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)

        self.log("ptl/train_loss", loss)
        self.log("ptl/train_accuracy", accuracy)
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
        self.eval_loss.clear()
        self.eval_accuracy.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=1, T_mult=30, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "ptl/val_loss",
        }


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.data_dir = "./data/"
        self.batch_size = config.batch_size
        # TODO: Add other transforms
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.num_workers = config.num_workers

    def setup(self, stage=None):
        with FileLock(f"{self.data_dir}.lock"):
            if config.dataet == "mnist":
                mnist = datasets.MNIST(
                    self.data_dir, train=True, download=True, transform=self.transform
                )
                self.data_train, self.data_val = random_split(mnist, [55000, 5000])

                self.data_test = datasets.MNIST(
                    self.data_dir, train=False, download=True, transform=self.transform
                )
            elif config.dataset == "emnist":
                mnist = datasets.EMNIST(
                    self.data_dir, train=True, download=True, transform=self.transform
                )
                # TODO: Check size of emnist
                self.data_train, self.data_val = random_split(mnist, [55000, 5000])
                raise ValueError("split not ready")
                self.data_test = datasets.EMNIST(
                    self.data_dir, train=False, download=True, transform=self.transform
                )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    config = gijo_config.read_config_from_file()
    pl.seed_everything(config.seed)

    dm = MNISTDataModule(batch_size=config.batch_size, num_workers=config.workers)

    model = MNISTClassifier(config)

    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        monitor="ptl/val_loss",
        dirpath=config.checkpoint_dir,
        filename=config.checkpoint_name_pattern,
        mode="min",
    )

    early_stopping_callback = pl_callbacks.EarlyStopping(
        monitor="ptl/val_loss", mode="min", patience=50
    )

    learning_rate_monitor = pl_callbacks.LearningRateMonitor(logging_interval="step")

    trainer_callbacks = [
        checkpoint_callback,
        learning_rate_monitor,
        early_stopping_callback,
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

    trainer.fit(model, datamodule=dm)
