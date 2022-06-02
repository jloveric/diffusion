# Original source https://github.com/acids-ircam/diffusion_models
from torch import Tensor
from inspect import isfunction
from typing import Any


def exists(x: Any):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num: int, divisor: int):
    """
    Split num int to groups of size divisor.  The remainder is placed
    in the last group.
    """
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def normalize_to_neg_one_to_one(img: Tensor):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t: Tensor):
    return (t + 1) * 0.5


def extract(a, t: Tensor, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
