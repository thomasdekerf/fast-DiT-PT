"""PyTorch Lightning training script for DiT.

This script wraps the existing DiT training logic into a LightningModule
so that we get richer logging, TensorBoard support and configurable
learning rate schedulers via a YAML config file.
"""

import argparse
from copy import deepcopy
import os
import yaml

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

# Reuse utility functions and dataset from the original training script
from train import CustomDataset, update_ema, requires_grad


class DiTLightningModule(pl.LightningModule):
    """LightningModule encapsulating a DiT model and diffusion training."""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        latent_size = cfg["image_size"] // 8
        self.model = DiT_models[cfg["model"]](
            input_size=latent_size, num_classes=cfg.get("num_classes", 1000)
        )
        self.diffusion = create_diffusion(timestep_respacing="")
        self.vae = AutoencoderKL.from_pretrained(
            f"stabilityai/sd-vae-ft-{cfg.get('vae', 'ema')}"
        )

        # Expose the training type for logging purposes
        self.learning_type = cfg.get("learning_type", "diffusion-transformer")

        # Exponential moving average of model weights
        self.ema = deepcopy(self.model)
        requires_grad(self.ema, False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["optimizer"]["lr"])
        sched_cfg = self.hparams.get("scheduler")
        if sched_cfg:
            name = sched_cfg.get("name", "").lower()
            if name == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=sched_cfg.get("T_max", self.trainer.max_epochs),
                    eta_min=sched_cfg.get("eta_min", 0.0),
                )
                return {"optimizer": optimizer, "lr_scheduler": scheduler}
            if name == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=sched_cfg.get("step_size", 1),
                    gamma=sched_cfg.get("gamma", 0.1),
                )
                return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), device=self.device)
        model_kwargs = dict(y=y)
        loss_dict = self.diffusion.training_losses(self.model, x, t, model_kwargs)
        loss = loss_dict["loss"].mean()

        for k, v in loss_dict.items():
            self.log(k, v.mean(), on_step=True, prog_bar=(k == "loss"))
        # Log current learning rate
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, on_step=True, prog_bar=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Update EMA weights after each optimization step
        update_ema(self.ema, self.model)

    def on_train_start(self):
        # Record the learning type in TensorBoard for experiment tracking
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_text("learning/type", self.learning_type)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(cfg_path):
    cfg = load_config(cfg_path)

    features_dir = os.path.join(cfg["feature_path"], "imagenet256_features")
    labels_dir = os.path.join(cfg["feature_path"], "imagenet256_labels")
    dataset = CustomDataset(features_dir, labels_dir)

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    model = DiTLightningModule(cfg)
    logger = TensorBoardLogger(save_dir=cfg["results_dir"], name="dit")
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"],
        logger=logger,
        callbacks=[lr_monitor],
        log_every_n_steps=cfg.get("log_every", 50),
    )
    trainer.fit(model, loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DiT with PyTorch Lightning")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)

