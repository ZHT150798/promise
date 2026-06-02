from collections.abc import Callable
from typing import Any

DatasetBuilder = Callable[..., Any]
_DATASETS: dict[str, DatasetBuilder] = {}


def register_dataset(name: str, builder: DatasetBuilder) -> None:
    if name in _DATASETS:
        raise ValueError(f"Dataset already registered: {name}")
    _DATASETS[name] = builder


def build_dataset(name: str, **kwargs: Any) -> Any:
    try:
        builder = _DATASETS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset: {name}") from exc
    return builder(**kwargs)


def _build_wsi_sr_dataset(**kwargs: Any) -> Any:
    from .wsi_sr_dataset import WsiSRDataset

    return WsiSRDataset(**kwargs)


def _build_stub_wsi_dataset(**kwargs: Any) -> Any:
    from .stub_dataset import build_stub_wsi_dataset

    return build_stub_wsi_dataset(**kwargs)


register_dataset("wsi_sr", _build_wsi_sr_dataset)
register_dataset("stub_wsi_sr", _build_stub_wsi_dataset)

__all__ = ["build_dataset", "register_dataset"]
