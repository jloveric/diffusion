# modified from https://github.com/lucidrains/denoising-diffusion-pytorch)

import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm
from torch import Tensor
from .util import (
    default,
    extract,
)


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
