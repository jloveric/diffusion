from diffusion.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

from torchvision import datasets
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
import math

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
