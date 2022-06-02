import torch

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from diffusion.exponential_moving_average import EMA
from pytorch_lightning.callbacks import Callback

from torchvision.utils import save_image
from diffusion.util import num_to_groups, unnormalize_to_zero_to_one
from torchvision.utils import make_grid
import logging

logger = logging.getLogger(__name__)


class ImageSampler(pl.callbacks.Callback):
    def __init__(
        self,
        ema_model: torch.nn.Module,
        batch_size: int = 36,
        samples: int = 36,
        directory: str = None,
    ) -> None:
        super().__init__()
        self._ema_model = ema_model
        self._batch_size = batch_size
        self._samples = samples
        self._directory = directory

    @rank_zero_only
    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._ema_model.eval()

        # Create a list of equal batch sizes with the last element possibly being smaller
        batches = num_to_groups(self._samples, self._batch_size)

        # Create a bunch of denoised samples (images) from the exponential moving average model
        all_images_list = list(
            map(lambda n: self._ema_model.sample(batch_size=n), batches)
        )

        all_images = torch.cat(all_images_list, dim=0)

        # TODO: make sure this is denormalized properly
        all_images = unnormalize_to_zero_to_one(all_images)

        img = make_grid(all_images).permute(1, 2, 0).cpu().numpy()

        # PLOT IMAGES
        trainer.logger.experiment.add_image(
            "img", torch.tensor(img).permute(2, 0, 1), global_step=trainer.global_step
        )

        """
        torchvision.utils.save_image(
            all_images,
            f"{self._directory}/results/sample-{trainer.global_step}.png",
            nrow=6,
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
        """
        Exponential moving average callback to be used with lightning trainer
        Args :
            ema : The exponential moving average class instance
            ema_model : The model containing the exponential moving average
            step_start_ema : Start computing the average after this number of steps
            update_ema_every : Update the ema after this number of steps.
        """
        self._ema = ema
        self._ema_model = ema_model
        self._step_start_ema = step_start_ema
        self._update_ema_every = update_ema_every
        self._started = False

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        if trainer.global_step >= self._step_start_ema and self._started is False:
            self._started = True
            self._ema.copy_current(self._ema_model, pl_module.model)
            logger.info("EMA copied base model")
        elif (
            trainer.global_step >= self._step_start_ema
            and trainer.global_step % self._update_ema_every == 0
        ):
            self._ema.update_model_average(self._ema_model, pl_module.model)


class Diffusion(pl.LightningModule):
    def __init__(self, diffusion_model, train_lr: float = 2e-5, amp: bool = False):
        """
        Args :
            diffusion_model :
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = diffusion_model
        self.train_lr = train_lr
        self.step = 0
        self.amp = amp
        # self.scaler = GradScaler(enabled=amp)

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
