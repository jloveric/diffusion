import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
from .exponential_moving_average import EMA
from torch.utils import data
from torch.cuda.amp import autocast, GradScaler

from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from PIL import Image

from tqdm import tqdm
from einops import rearrange
from torch import Tensor
from .util import exists, default


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self, denoise_fn, *, image_size, channels=3, timesteps=1000, loss_type="l1"
    ):
        super().__init__()
        assert not (
            type(self) == GaussianDiffusion
            and denoise_fn.channels != denoise_fn.out_dim
        )

        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn

        betas = cosine_beta_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Remove the noise from the image for 1 timestep back.
        The predicted noise is computed from the model.
        Args :
            x_t : x at time t
            t : time
            noise : The noise predicted by the model at the given time.
        Return :
            x_{t-1}
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """
        Compute gaussian quantites for q(x_t|x_{t-1}) I believe!
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x: Tensor, t: Tensor, clip_denoised: bool):
        """
        Compute mean and variance for p(x_{t-1}|x_t)
        Args :
            x : images
            t : time at each batch
            clip_denoised : When the noise is removed the data can be out of [-1,1] so
            the data can be clipped when it's out of range.
        """

        # Compute x_{t-1}
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t))

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self, x: Tensor, t: Tensor, clip_denoised: bool = True, repeat_noise=False
    ):
        """
        Args :
            x : Tensor of images
            t : 1d Tensor of times
        """
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised
        )
        noise = noise_like(x.shape, device, repeat_noise)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        # Return the image with noise from 1 step removed
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        """
        Create samples by running through the complete reverse process.
        Args :
            shape : number of batches and shape of the image to create. (b, c, h, w)
        """
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            # You need to denoise num_timesteps times to get back to
            # the predicted image.
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long)
            )
        return img

    @torch.no_grad()
    def sample(self, batch_size=16):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size))

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(
            reversed(range(0, t)), desc="interpolation sample time step", total=t
        ):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long)
            )

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")

    def p_losses(self, x_start, t, noise=None):
        b, c, h, w = x_start.shape

        # return noise if it's defined otherwise return random value
        # in the shape of x_start
        noise = default(noise, lambda: torch.randn_like(x_start))

        # compute the noised version of x_start at time t
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # run through the denoising NN
        # Given P(t) compute P(t-1) - well actually compute the noise component from the noised image
        x_recon = self.denoise_fn(x_noisy, t)

        # And if we compute the noise correctly this should be the loss
        loss = self.loss_fn(noise, x_recon)
        return loss

    def forward(self, x, *args, **kwargs):
        b, c, h, w, device, img_size, = (
            *x.shape,
            x.device,
            self.image_size,
        )
        assert (
            h == img_size and w == img_size
        ), f"height and width of image must be {img_size} but got {h} and {w}"

        # grab some random timesteps in the diffusion process
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(x, t, *args, **kwargs)


# dataset classes


class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts=["jpg", "jpeg", "png", "JPEG"]):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f"{folder}").glob(f"**/*.{ext}")]

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Lambda(normalize_to_neg_one_to_one),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


# trainer class
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        ema_decay=0.995,
        image_size=128,
        train_batch_size=32,
        train_lr=2e-5,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder="./results",
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        # apparently we don't start the ema until the model has
        # been trained a while.
        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.ds = Dataset(folder, image_size)
        self.dl = cycle(
            data.DataLoader(
                self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True
            )
        )
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        torch.save(data, str(self.results_folder / f"model-{milestone}.pt"))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f"model-{milestone}.pt"))

        self.step = data["step"]
        self.model.load_state_dict(data["model"])
        self.ema_model.load_state_dict(data["ema"])
        self.scaler.load_state_dict(data["scaler"])

    def train(self):
        while self.step < self.train_num_steps:

            # accumulate gradients over n steps
            for i in range(self.gradient_accumulate_every):

                # grab the next data from the dataloader
                data = next(self.dl).cuda()

                # autocast allows things to run in mixed precision
                with autocast(enabled=self.amp):

                    # train the denoising model, typically GaussianDiffusion!
                    loss = self.model(data)
                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                print(f"{self.step}: {loss.item()}")

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            # Periodically update the exponential moving average given the diffusion model
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                self.ema_model.eval()

                milestone = self.step // self.save_and_sample_every

                # Create a list of equal batch sizes with the last element possibly being smaller
                batches = num_to_groups(36, self.batch_size)

                # Create a bunch of denoised samples (images) from the exponential moving average model
                all_images_list = list(
                    map(lambda n: self.ema_model.sample(batch_size=n), batches)
                )

                all_images = torch.cat(all_images_list, dim=0)
                all_images = unnormalize_to_zero_to_one(all_images)

                # save all output in one giant combined image
                utils.save_image(
                    all_images,
                    str(self.results_folder / f"sample-{milestone}.png"),
                    nrow=6,
                )
                self.save(milestone)

            self.step += 1

        print("training completed")
