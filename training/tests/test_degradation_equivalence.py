import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as torch_f
from PIL import Image


pytest.importorskip("basicsr")

from dataloaders.realesrgan import RealESRGAN_degradation
from promise_train.data.degradation.realesrgan import RealESRGANWSIDegradation
from promise_train.data.wsi_sr_dataset import WsiSRDataset


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_wsi_compress_wrapper_matches_reference() -> None:
    seed = 42
    rng = np.random.default_rng(seed)
    img_gt = rng.random((1024, 1024, 3), dtype=np.float32)
    img_lr = rng.random((256, 256, 3), dtype=np.float32)

    old = RealESRGAN_degradation("params_realesrgan.yml", device="cpu")
    new = RealESRGANWSIDegradation("dataloaders/params_realesrgan.yml", device="cpu")

    _seed_all(seed)
    old_gt, old_lr, old_prompt, old_mag_prompt = old.random_augment_wsi_compress(
        img_gt.copy(), img_lr.copy()
    )

    _seed_all(seed)
    new_gt, new_lr, new_prompt, new_mag_prompt = new.random_augment_wsi_compress(
        img_gt.copy(), img_lr.copy()
    )

    assert old_prompt == new_prompt
    assert old_mag_prompt == new_mag_prompt
    torch.testing.assert_close(old_gt, new_gt, rtol=0, atol=0)
    torch.testing.assert_close(old_lr, new_lr, rtol=0, atol=0)


def test_multi_wrapper_matches_reference() -> None:
    seed = 42
    rng = np.random.default_rng(seed)
    img_gt = rng.random((1024, 1024, 3), dtype=np.float32)
    img_lr = rng.random((256, 256, 3), dtype=np.float32)

    old = RealESRGAN_degradation("params_realesrgan.yml", device="cpu")
    new = RealESRGANWSIDegradation("dataloaders/params_realesrgan.yml", device="cpu")

    _seed_all(seed)
    old_gt, old_lr, old_prompt = old.random_augment_multi(img_gt.copy(), img_lr.copy(), mag="20x")

    _seed_all(seed)
    new_gt, new_lr, new_prompt = new.random_augment_multi(img_gt.copy(), img_lr.copy(), mag="20x")

    assert old_prompt == new_prompt
    torch.testing.assert_close(old_gt, new_gt, rtol=0, atol=0)
    torch.testing.assert_close(old_lr, new_lr, rtol=0, atol=0)


def _image_from_array(array: np.ndarray) -> Image.Image:
    return Image.fromarray((array * 255.0).round().clip(0, 255).astype(np.uint8))


def _dataset_shell(mag: str = "40x", scale: int = 4) -> WsiSRDataset:
    dataset = object.__new__(WsiSRDataset)
    dataset.split = "test"
    dataset.data_cfg = SimpleNamespace()
    dataset.mag = mag
    dataset.scale = scale
    dataset.deg_type = "wsisr_compress"
    dataset.degradation = RealESRGANWSIDegradation("dataloaders/params_realesrgan.yml", device="cpu")
    return dataset


def test_val_1024_path_matches_reference_random_augment_multi() -> None:
    rng = np.random.default_rng(7)
    img_gt = rng.integers(0, 256, size=(1024, 1024, 3), dtype=np.uint8).astype(np.float32) / 255.0
    img_lr = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8).astype(np.float32) / 255.0
    dataset = _dataset_shell(mag="40x", scale=4)
    old = RealESRGAN_degradation("params_realesrgan.yml", device="cpu")

    old_gt, old_lr, old_prompt = old.random_augment_multi(img_gt.copy(), img_lr.copy(), mag="40x")
    new_gt, new_lr, new_prompt = dataset._augment_val_wsi_compress(
        _image_from_array(img_gt),
        _image_from_array(img_lr),
    )

    assert old_prompt == new_prompt
    torch.testing.assert_close(old_gt, new_gt, rtol=0, atol=0)
    torch.testing.assert_close(old_lr, new_lr, rtol=0, atol=0)


def test_val_512_passthrough_keeps_512_and_40x_pe_padding_is_legal() -> None:
    rng = np.random.default_rng(8)
    img_gt = rng.random((512, 512, 3), dtype=np.float32)
    img_lr = rng.random((128, 128, 3), dtype=np.float32)
    dataset = _dataset_shell(mag="40x", scale=4)

    gt_t, lr_t, prompt = dataset._augment_val_wsi_compress(
        _image_from_array(img_gt),
        _image_from_array(img_lr),
    )

    assert prompt == "40× magnification, 4× super-resolution"
    assert gt_t.shape == (1, 3, 512, 512)
    assert lr_t.shape == (1, 3, 512, 512)
    padded = torch_f.pad(gt_t, (256, 256, 256, 256), mode="reflect")
    assert padded.shape == (1, 3, 1024, 1024)


def test_val_unsupported_size_fails_loudly() -> None:
    rng = np.random.default_rng(9)
    dataset = _dataset_shell(mag="40x", scale=4)

    with pytest.raises(ValueError, match="expected 512 or 1024"):
        dataset._augment_val_wsi_compress(
            _image_from_array(rng.random((768, 768, 3), dtype=np.float32)),
            _image_from_array(rng.random((192, 192, 3), dtype=np.float32)),
        )


def test_train_max_samples_limits_train_file_list(tmp_path) -> None:
    hr_dir = tmp_path / "hr"
    lr_dir = tmp_path / "lr"
    hr_dir.mkdir()
    lr_dir.mkdir()
    for idx in range(5):
        (hr_dir / f"{idx}.png").write_bytes(b"placeholder")
        (lr_dir / f"{idx}.png").write_bytes(b"placeholder")

    dataset = WsiSRDataset(
        split="train",
        data_cfg=SimpleNamespace(
            train_path=str(tmp_path),
            val_path=str(tmp_path),
            deg_type="wsisr_compress",
            deg_file_path="dataloaders/params_realesrgan.yml",
            train_max_samples=2,
        ),
    )

    assert len(dataset) == 2
