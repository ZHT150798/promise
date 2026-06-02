from pathlib import Path
from typing import Any

import torch
from torch import nn

from .morphodiff import CHECKPOINT_KEYS, PAPER_RANK_UNET, PAPER_RANK_VAE


class _AdapterBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_weight = nn.Parameter(torch.tensor(0.1))
        self.conv_in = nn.Conv2d(3, 3, kernel_size=1)

    def set_adapter(self, _names: list[str]) -> None:
        return None

    def enable_xformers_memory_efficient_attention(self) -> None:
        return None

    def enable_gradient_checkpointing(self) -> None:
        return None


class _VAEBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_weight = nn.Parameter(torch.tensor(0.2))

    def set_adapter(self, _names: list[str]) -> None:
        return None


class StubMorphoDiffGen(nn.Module):
    def __init__(self, args: Any) -> None:
        super().__init__()
        self.args = args
        self.unet = _AdapterBlock()
        self.vae = _VAEBlock()

    def set_train(self) -> None:
        self.train()

    def forward(self, x_src, batch=None, prompt=None, args=None):
        pred = self.unet.conv_in(x_src) + self.unet.lora_weight + self.vae.lora_weight
        return pred, None, None

    def save_model(self, path: str) -> None:
        checkpoint = {key: {} for key in CHECKPOINT_KEYS}
        checkpoint["rank_unet"] = PAPER_RANK_UNET
        checkpoint["rank_vae"] = PAPER_RANK_VAE
        checkpoint["state_dict_unet"] = self.unet.state_dict()
        checkpoint["state_dict_vae"] = self.vae.state_dict()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)


def build_stub_morphodiff_gen(args: Any):
    return StubMorphoDiffGen(args)


__all__ = ["StubMorphoDiffGen", "build_stub_morphodiff_gen"]
