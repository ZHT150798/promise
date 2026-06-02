import torch
from torch.utils.data import Dataset


class StubWSIDataset(Dataset):
    def __init__(self, split: str = "train", data_cfg=None, **_kwargs) -> None:
        self.split = split
        self.length = 4 if split == "train" else 2

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        value = float(idx + 1)
        return {
            "conditioning_pixel_values": torch.ones(3, 8, 8) * value,
            "output_pixel_values": torch.ones(3, 8, 8) * (value + 0.5),
            "pos_prompt": "stub positive",
            "neg_prompt": "stub negative",
            "mag_prompt": "40× magnification",
        }


def build_stub_wsi_dataset(**kwargs):
    return StubWSIDataset(**kwargs)


__all__ = ["StubWSIDataset", "build_stub_wsi_dataset"]
