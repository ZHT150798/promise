from typing import Any, Optional

import torch
from torch import nn


class StubDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))

    def forward(self, x, for_G: bool = False, for_real: Optional[bool] = None):
        sign = 1.0 if (for_G or for_real) else -1.0
        return x.flatten(1).mean(dim=1) * self.weight * sign


def build_discriminator(loss_cfg: Any, device: str = "cuda"):
    if loss_cfg.gan_disc_type == "stub":
        net_disc = StubDiscriminator().to(device)
        net_disc.requires_grad_(True)
        net_disc.train()
        return net_disc

    if loss_cfg.gan_disc_type != "vagan":
        raise NotImplementedError(f"Discriminator type {loss_cfg.gan_disc_type} not implemented")

    import vision_aided_loss

    net_disc = vision_aided_loss.Discriminator(
        cv_type="dino",
        output_type="conv_multi_level",
        loss_type=loss_cfg.gan_loss_type,
        device=device,
    )
    net_disc = net_disc.to(device)
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()
    return net_disc


__all__ = ["StubDiscriminator", "build_discriminator"]
