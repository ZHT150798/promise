from dataclasses import dataclass

import torch

from dataloaders.realesrgan import DiffJPEG, RealESRGAN_degradation, USMSharp, opt_parse


@dataclass(frozen=True)
class RealESRGANWSIConfig:
    file_path: str
    device: str = "cpu"


class RealESRGANWSIDegradation:
    """Configurable wrapper around the shared Real-ESRGAN implementation."""

    def __init__(self, file_path: str, device: str = "cpu") -> None:
        self.impl = self._build_impl(file_path, device=device)

    def _build_impl(self, file_path: str, device: str):
        impl = object.__new__(RealESRGAN_degradation)
        impl.opt = opt_parse(file_path)
        impl.device = device
        optk = impl.opt["kernel_info"]

        impl.blur_kernel_size = optk["blur_kernel_size"]
        impl.kernel_list = optk["kernel_list"]
        impl.kernel_prob = optk["kernel_prob"]
        impl.blur_sigma = optk["blur_sigma"]
        impl.betag_range = optk["betag_range"]
        impl.betap_range = optk["betap_range"]
        impl.sinc_prob = optk["sinc_prob"]

        impl.blur_kernel_size2 = optk["blur_kernel_size2"]
        impl.kernel_list2 = optk["kernel_list2"]
        impl.kernel_prob2 = optk["kernel_prob2"]
        impl.blur_sigma2 = optk["blur_sigma2"]
        impl.betag_range2 = optk["betag_range2"]
        impl.betap_range2 = optk["betap_range2"]
        impl.sinc_prob2 = optk["sinc_prob2"]

        impl.final_sinc_prob = optk["final_sinc_prob"]
        impl.kernel_range = [2 * v + 1 for v in range(3, 11)]
        impl.pulse_tensor = torch.zeros(21, 21).float()
        impl.pulse_tensor[10, 10] = 1
        impl.jpeger = DiffJPEG(differentiable=False).to(impl.device)
        impl.usm_shaper = USMSharp().to(impl.device)
        return impl

    def degrade_process(self, *args, **kwargs):
        return self.impl.degrade_process(*args, **kwargs)

    def random_augment_pair(self, *args, **kwargs):
        return self.impl.random_augment_pair(*args, **kwargs)

    def random_augment_lr_compress(self, *args, **kwargs):
        return self.impl.random_augment_lr_compress(*args, **kwargs)

    def random_augment_wsi_compress(self, *args, **kwargs):
        return self.impl.random_augment_wsi_compress(*args, **kwargs)

    def random_augment_pair_infer(self, *args, **kwargs):
        return self.impl.random_augment_pair_infer(*args, **kwargs)

    def random_augment_multi(self, *args, **kwargs):
        return self.impl.random_augment_multi(*args, **kwargs)

    def random_augment_norm(self, *args, **kwargs):
        return self.impl.random_augment_norm(*args, **kwargs)


__all__ = ["RealESRGANWSIConfig", "RealESRGANWSIDegradation"]
