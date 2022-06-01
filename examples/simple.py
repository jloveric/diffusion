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
        timesteps=1000,
        loss_type="l1",  # number of steps  # L1 or L2
    ).cuda()

    image_datamodule = ImageDataModule(
        image_size=image_size,
        folder=cfg.data.folder,
        batch_size=32,
        num_workers=10,
        split_frac=0.8,
    )
    model = Diffusion(diffusion_model=diffusion)

    # Set up the exponential moving average.
    # Apparently diffusion models give better results if we
    # using a moving average of the weights.
    ema = EMA(beta=0.995)
    ema_model = copy.deepcopy(model)
    ema_callback = EMACallback(ema=ema, ema_model=ema_model)

    sampler = ImageSampler()
    logger = TensorBoardLogger("tb_logs", name="vae")
    trainer = Trainer(
        max_epochs=cfg.epochs,
        gpus=cfg.gpus,
        logger=logger,
        callbacks=[sampler, ema_callback],
    )

    trainer.fit(model, datamodule=image_datamodule)
    result = trainer.test(model)

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
