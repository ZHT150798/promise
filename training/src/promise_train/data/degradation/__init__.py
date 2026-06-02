from collections.abc import Callable
from typing import Any

from .realesrgan import RealESRGANWSIDegradation

DegradationBuilder = Callable[..., Any]
_DEGRADATIONS: dict[str, DegradationBuilder] = {}


def register_degradation(name: str, builder: DegradationBuilder) -> None:
    if name in _DEGRADATIONS:
        raise ValueError(f"Degradation already registered: {name}")
    _DEGRADATIONS[name] = builder


def build_degradation(name: str, **kwargs: Any) -> Any:
    try:
        builder = _DEGRADATIONS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown degradation: {name}") from exc
    return builder(**kwargs)


register_degradation("realesrgan_wsi", RealESRGANWSIDegradation)

__all__ = ["RealESRGANWSIDegradation", "build_degradation", "register_degradation"]
