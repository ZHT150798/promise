import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

from .degradation import RealESRGANWSIDegradation


class WsiSRDataset(torch.utils.data.Dataset):
    """WSI super-resolution dataset for training and validation."""

    def __init__(
        self,
        split: str,
        data_cfg: Any,
        mag: str = "40x",
        scale: int = 4,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.split = split
        self.data_cfg = data_cfg
        self.mag = mag
        self.scale = scale
        self.deg_type = data_cfg.deg_type
        self.degradation = RealESRGANWSIDegradation(data_cfg.deg_file_path, device="cpu")

        if split == "train":
            self.gt_path = os.path.join(data_cfg.train_path, "hr/")
            self.lr_path = os.path.join(data_cfg.train_path, "lr/")
        else:
            self.gt_path = os.path.join(data_cfg.val_path, "hr/")
            lr_dir = f"lr_X{scale}/" if scale != 4 else "lr/"
            self.lr_path = os.path.join(data_cfg.val_path, lr_dir)

        self.gt_list = sorted(str(x) for x in Path(self.gt_path).rglob("*.png"))
        if split == "train" and getattr(data_cfg, "train_max_samples", None) is not None:
            self.gt_list = self.gt_list[: int(data_cfg.train_max_samples)]
        if split != "train" and max_samples is not None:
            self.gt_list = self.gt_list[:max_samples]

    def __len__(self) -> int:
        return len(self.gt_list)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_name = os.path.basename(self.gt_list[idx])
        gt_img = Image.open(os.path.join(self.gt_path, img_name)).convert("RGB")
        lr_img = self._load_lr_image(img_name) if self.deg_type in {"lr", "compress", "wsisr_compress"} else None

        sr_prompt = ""
        mag_prompt = None

        if self.deg_type == "realgan":
            output_t, img_t = self.degradation.degrade_process(np.asarray(gt_img) / 255.0, resize_bak=True)
        elif self.deg_type == "lr":
            output_t, img_t = self.degradation.random_augment_pair(
                np.asarray(gt_img) / 255.0, np.asarray(lr_img) / 255.0
            )
        elif self.deg_type == "compress":
            if self.split == "train":
                output_t, img_t, sr_prompt = self.degradation.random_augment_lr_compress(
                    np.asarray(gt_img) / 255.0, np.asarray(lr_img) / 255.0
                )
            else:
                output_t, img_t, sr_prompt = self.degradation.random_augment_pair_infer(
                    np.asarray(gt_img) / 255.0, np.asarray(lr_img) / 255.0
                )
        elif self.deg_type == "wsisr_compress":
            if self.split == "train":
                output_t, img_t, sr_prompt, mag_prompt = self.degradation.random_augment_wsi_compress(
                    np.asarray(gt_img) / 255.0, np.asarray(lr_img) / 255.0
                )
            else:
                output_t, img_t, sr_prompt = self._augment_val_wsi_compress(gt_img, lr_img)
        elif self.deg_type == "norm":
            output_t, img_t = self.degradation.random_augment_norm(np.asarray(gt_img) / 255.0)
        else:
            raise ValueError(f"Incorrect deg type {self.deg_type}")

        output_t = output_t.squeeze(0)
        img_t = img_t.squeeze(0)
        img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        example: dict[str, Any] = {
            "neg_prompt": sr_prompt,
            "pos_prompt": sr_prompt,
            "null_prompt": "",
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
        }
        if mag_prompt is not None:
            example["mag_prompt"] = mag_prompt
        return example

    def _load_lr_image(self, img_name: str) -> Image.Image:
        if self.split == "train" or self.deg_type != "wsisr_compress":
            return Image.open(os.path.join(self.lr_path, img_name)).convert("RGB")

        img_base = os.path.splitext(img_name)[0]
        png_path = os.path.join(self.lr_path, img_base + ".png")
        jpg_path = os.path.join(self.lr_path, img_base + ".jpg")
        if os.path.exists(png_path):
            return Image.open(png_path).convert("RGB")
        if os.path.exists(jpg_path):
            return Image.open(jpg_path).convert("RGB")
        raise FileNotFoundError(f"Neither {png_path} nor {jpg_path} exists.")

    def _augment_val_wsi_compress(self, gt_img: Image.Image, lr_img: Image.Image):
        width, height = gt_img.size
        if width != height:
            raise ValueError(f"Validation WSI patch must be square, got {gt_img.size}.")
        if height == 1024:
            return self.degradation.random_augment_multi(
                np.asarray(gt_img) / 255.0, np.asarray(lr_img) / 255.0, mag=self.mag
            )
        if height != 512:
            raise ValueError(f"Unsupported validation WSI patch size {gt_img.size}; expected 512 or 1024.")

        img_gt = torch.from_numpy(np.asarray(gt_img) / 255.0).permute(2, 0, 1).float().unsqueeze(0)
        img_lr = torch.from_numpy(np.asarray(lr_img) / 255.0).permute(2, 0, 1).float().unsqueeze(0)
        img_lr = torch.nn.functional.interpolate(img_lr, size=(height, width), mode="bicubic", align_corners=False)
        img_lr = torch.clamp((img_lr * 255.0).round(), 0, 255) / 255.0

        if self.mag == "40x":
            mag_prompt = "40× magnification"
        elif self.mag == "20x":
            mag_prompt = "20× magnification"
        else:
            raise ValueError(f"Unsupported magnification: {self.mag}")

        if self.scale == 4:
            scale_prompt = "4× super-resolution"
        elif self.scale == 8:
            scale_prompt = "8× super-resolution"
        else:
            raise ValueError(f"Unsupported scale_factor: {self.scale}")

        return img_gt, img_lr, f"{mag_prompt}, {scale_prompt}"


__all__ = ["WsiSRDataset"]
