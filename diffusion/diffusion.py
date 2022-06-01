from typing import Optional, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from .util import normalize_to_neg_one_to_one
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from diffusion.exponential_moving_average import EMA
from torch.cuda.amp import autocast, GradScaler
from pytorch_lightning.callbacks import Callback
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from diffusion.denoising_diffusion_pytorch import GaussianDiffusion
from diffusion.unet import Unet
import copy
from torch.utils.data import Dataset
from PIL import Image
from diffusion.util import num_to_groups, unnormalize_to_zero_to_one
from io import BytesIO
from torchvision.utils import make_grid


class ImageSampler(pl.callbacks.Callback):
    def __init__(self, ema_model: torch.nn.Module, batch_size=32) -> None:
        super().__init__()
        self._ema_model = ema_model
        self._batch_size = batch_size

    @rank_zero_only
    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._ema_model.eval()

        # Create a list of equal batch sizes with the last element possibly being smaller
        batches = num_to_groups(36, self._batch_size)

        # Create a bunch of denoised samples (images) from the exponential moving average model
        all_images_list = list(
            map(lambda n: self._ema_model.sample(batch_size=n), batches)
        )

        all_images = torch.cat(all_images_list, dim=0)

        # TODO: make sure this is denormalized properly
        all_images = unnormalize_to_zero_to_one(all_images) * 256

        img = make_grid(all_images).permute(1, 2, 0).cpu().numpy()

        # PLOT IMAGES
        trainer.logger.experiment.add_image(
            "img", torch.tensor(img).permute(2, 0, 1), global_step=trainer.global_step
        )

        """
        iobuf = BytesIO()

        # save all output in one giant combined image
        torchvision.utils.save_image(
            tensor=all_images,
            fp=iobuf,
            nrow=6,
        )

        trainer.logger.experiment.add_image(
            "img", iobuf, global_step=trainer.global_step
        )
        """


class EMACallback(Callback):
    def __init__(
        self,
        ema: EMA,
        ema_model: torch.nn.Module,
        step_start_ema: int = 2000,
        update_ema_every: int = 10,
    ):
        self._ema = ema
        self._ema_model = ema_model
        self._step_start_ema = step_start_ema
        self._update_ema_every = update_ema_every

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        if (
            trainer.global_step >= self._step_start_ema
            and trainer.global_step % self._update_ema_every == 0
        ):
            self._ema.update_model_average(self._ema_model, pl_module.model)


class Diffusion(pl.LightningModule):
    def __init__(
        self,
        diffusion_model,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        amp=False,
        step_start_ema=2000,
        update_ema_every=10,
    ):
        """
        Args :
            diffusion_model :
        """
        super().__init__()

        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        # self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        # apparently we don't start the ema until the model has
        # been trained a while.
        self.step_start_ema = step_start_ema

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.train_lr = train_lr

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)

        # self.reset_parameters()

    def forward(self, x):
        # the model returns the loss instead of the
        # evaluation.  The evaluation is actually the
        # predicted noise, so not very meaningful.
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # x, y = batch you don't need this since the
        # target is generated internally by diffusing the
        # model.  As a result x=batch!
        loss = self(batch)
        self.log(f"train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.log(f"val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log(f"test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.train_lr)
        return optimizer
