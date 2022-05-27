from diffusion.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from diffusion.diffusion import *

from torchvision import datasets
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
import math
from pl_examples import _DATASETS_PATH, cli_lightning_logo
from pl_examples.basic_examples.mnist_datamodule import MNIST

"""
train_data = datasets.MNIST(
    root="data",
    train=True,
    transform=ToTensor(),
    download=True,
)
test_data = datasets.MNIST(root="data", train=False, transform=ToTensor())
"""

image_size = 32

model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).cuda()


diffusion = GaussianDiffusion(
    model,
    image_size=image_size,
    timesteps=1000,
    loss_type="l1",  # number of steps  # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    "/mnt/1000gb/cifar10/test/airplane",
    image_size=image_size,
    train_batch_size=32,
    train_lr=2e-5,
    train_num_steps=1000000,  # total training steps
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=True,  # turn on mixed precision
)

trainer.train()

sampled_images = diffusion.sample(batch_size=4)
width = int(math.sqrt(len(sampled_images)))
height = int(math.ceil(len(sampled_images) / width))

f, axarr = plt.subplots(width, height)
axarr = axarr.flatten()
for index, e in enumerate(sampled_images):
    print(axarr[index])
    aval = e.permute(1, 2, 0).cpu().detach().numpy()
    axarr[index].imshow(aval)

plt.show()


class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        dataset = MNIST(
            _DATASETS_PATH, train=True, download=True, transform=transforms.ToTensor()
        )
        self.mnist_test = MNIST(
            _DATASETS_PATH, train=False, download=True, transform=transforms.ToTensor()
        )
        self.mnist_train, self.mnist_val = random_split(dataset, [55000, 5000])
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


def cli_main():
    cli = LightningCLI(
        LitAutoEncoder,
        MyDataModule,
        seed_everything_default=1234,
        save_config_overwrite=True,
        run=False,  # used to de-activate automatic fitting.
        trainer_defaults={"callbacks": ImageSampler(), "max_epochs": 10},
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
    predictions = cli.trainer.predict(ckpt_path="best", datamodule=cli.datamodule)
    print(predictions[0])


if __name__ == "__main__":
    cli_lightning_logo()
    cli_main()
