from diffusion.diffusion import Diffusion, EMACallback, ImageSampler
from diffusion.denoising_diffusion_pytorch import GaussianDiffusion
from diffusion.unet import Unet
import hydra
from omegaconf import DictConfig, OmegaConf
from diffusion.data import ImageDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningModule, Trainer, Callback
from diffusion.exponential_moving_average import EMA
import copy


def train_diffusion(cfg: DictConfig):

    image_size = cfg.data.image_size

    diffusion = GaussianDiffusion(
        denoise_fn=Unet(dim=64, dim_mults=(1, 2, 4, 8)).cuda(),
        image_size=image_size,
        timesteps=1000,  # number of steps in forward and reverse process
        loss_type="l1",  # number of steps  # L1 or L2
    ).cuda()

    image_datamodule = ImageDataModule(
        image_size=image_size,
        folder=cfg.data.folder,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        split_frac=cfg.data.split_frac,
    )
    model = Diffusion(diffusion_model=diffusion)

    # Set up the exponential moving average.
    # Diffusion models give better results if we
    # using a moving average of the weights.
    ema = EMA(beta=cfg.ema.beta)
    ema_model = copy.deepcopy(diffusion)
    ema_callback = EMACallback(ema=ema, ema_model=ema_model)

    sampler = ImageSampler(ema_model=ema_model)
    logger = TensorBoardLogger("tb_logs", name="diffusion")
    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        max_steps=cfg.max_steps,
        gpus=cfg.gpus,
        logger=logger,
        callbacks=[sampler, ema_callback],
    )

    trainer.fit(model, datamodule=image_datamodule)
    result = trainer.test(model, datamodule=image_datamodule)

    print("finished testing")
    print("result", result)
    return result


@hydra.main(config_path="../configs", config_name="simple")
def run(cfg: DictConfig):
    result = train_diffusion(cfg=cfg)

    # needed for nevergrad
    return result


if __name__ == "__main__":
    run()
