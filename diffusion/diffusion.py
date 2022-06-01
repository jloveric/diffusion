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


class ImageSampler(pl.callbacks.Callback):
    def __init__(
        self,
        num_samples: int = 3,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = True,
        norm_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
    ) -> None:
        """
        Args:
            num_samples: Number of images displayed in the grid. Default: ``3``.
            nrow: Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding: Amount of padding. Default: ``2``.
            normalize: If ``True``, shift the image to the range (0, 1),
                by the min and max values specified by :attr:`range`. Default: ``False``.
            norm_range: Tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each: If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value: Value for the padded pixels. Default: ``0``.
        """

        super().__init__()
        self.num_samples = num_samples
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value

    def _to_grid(self, images):
        return torchvision.utils.make_grid(
            tensor=images,
            nrow=self.nrow,
            padding=self.padding,
            normalize=self.normalize,
            range=self.norm_range,
            scale_each=self.scale_each,
            pad_value=self.pad_value,
        )

    @rank_zero_only
    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        images, _ = next(
            iter(DataLoader(trainer.datamodule.mnist_val, batch_size=self.num_samples))
        )
        images_flattened = images.view(images.size(0), -1)

        # generate images
        with torch.no_grad():
            pl_module.eval()
            images_generated = pl_module(images_flattened.to(pl_module.device))
            pl_module.train()

        if trainer.current_epoch == 0:
            save_image(self._to_grid(images), f"grid_ori_{trainer.current_epoch}.png")
        save_image(
            self._to_grid(images_generated.reshape(images.shape)),
            f"grid_generated_{trainer.current_epoch}.png",
        )
        """


class EMACallback(Callback):
    def __init__(self, ema: EMA, ema_model):
        self._ema = ema
        self._ema_model = ema_model

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
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
        self.ema_model = copy.deepcopy(self.model)
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
