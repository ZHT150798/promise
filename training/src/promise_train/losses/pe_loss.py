from typing import Any, Union

import torch
from torch import nn

from my_utils.fd_loss import pe_loss
from my_utils.fd_loss.conch import build_conch


class StubPELoss(nn.Module):
    def forward(self, pred, target, mag_prompt=None):
        return (pred - target).pow(2).mean() * 0.0


def build_pe_loss(loss_cfg: Any, device: Union[str, torch.device] = "cuda"):
    pe_type = loss_cfg.pe_type
    if pe_type == "stub":
        return StubPELoss().to(device)

    if pe_type == "conch":
        conch, _ = build_conch(getattr(loss_cfg, "conch_checkpoint", None))
        return pe_loss.PELoss(conch.to(device))

    if pe_type == "uni":
        import timm

        model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        model.load_state_dict(torch.load(loss_cfg.uni_checkpoint, map_location="cpu"), strict=True)
        return pe_loss.PELoss_uni(model.to(device))

    if pe_type == "gigapath":
        import timm

        model = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath",
            pretrained=False,
            checkpoint_path=loss_cfg.gigapath_checkpoint,
        )
        return pe_loss.PELoss_prov(model.to(device))

    raise ValueError(f"Unsupported pe_type: {pe_type}")


__all__ = ["StubPELoss", "build_pe_loss"]
