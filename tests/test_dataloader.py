import pytest
from diffusion.data import ImageDataModule


def test_image_dataset():
    data_module = ImageDataModule(
        image_size=32, folder="test_data", batch_size=1, num_workers=1, split_frac=0.8
    )
    data_module.setup()

    assert len(data_module.train_dataloader()) == 5
    assert len(data_module.test_dataloader()) == 1
    assert len(data_module.val_dataloader()) == 1
