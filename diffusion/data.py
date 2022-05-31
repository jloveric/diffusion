from typing import Optional, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

import pytorch_lightning as pl
from .util import normalize_to_neg_one_to_one
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.utilities.imports import _TORCHVISION_AVAILABLE
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from exponential_moving_average import EMA

if _TORCHVISION_AVAILABLE:
    import torchvision
    from torchvision import transforms
    from torchvision.utils import save_image
from .denoising_diffusion_pytorch import GaussianDiffusion
from .unet import Unet
import copy
from PIL import Image
from typing import List


class GenericImageDataset(Dataset):
    def __init__(self, image_size: int, path_list: List[str]):
        super().__init__()
        self.image_size = image_size
        self.paths = path_list

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


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_size: int,
        folder: str,
        exts: List[str] = ["jpg", "jpeg", "png", "JPEG"],
    ):
        super().__init__()
        self._image_size = image_size
        self._folder = folder
        self._exts = exts

    def setup(self, stage: Optional[str] = None):

        self.folder = self._folder
        self.image_size = self._image_size
        self.paths = [
            p for ext in self._exts for p in Path(f"{self.folder}").glob(f"**/*.{ext}")
        ]
        size = len(self.paths)

        train_size = int(0.9 * size)
        test_size = (size - test_size) // 2
        val_size = size - train_size - test_size

        self._train_list, self._test_list, self._val_list = random_split(
            self.paths, [train_size, test_size, val_size]
        )

    def train_dataloader(self):
        return GenericImageDataset(
            image_size=self._image_size, path_list=self._train_list
        )

    def val_dataloader(self):
        return GenericImageDataset(
            image_size=self._image_size, path_list=self._val_list
        )

    def test_dataloader(self):
        return GenericImageDataset(
            image_size=self._image_size, path_list=self._test_list
        )
