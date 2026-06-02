from collections.abc import Callable
from typing import Any

LossBuilder = Callable[..., Any]
_LOSSES: dict[str, LossBuilder] = {}


def register_loss(name: str, builder: LossBuilder) -> None:
    if name in _LOSSES:
        raise ValueError(f"Loss already registered: {name}")
    _LOSSES[name] = builder


def build_loss(name: str, **kwargs: Any) -> Any:
    try:
        builder = _LOSSES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown loss: {name}") from exc
    return builder(**kwargs)


def _build_pe_loss(**kwargs: Any) -> Any:
    from .pe_loss import build_pe_loss

    return build_pe_loss(**kwargs)


register_loss("pe", _build_pe_loss)

__all__ = ["build_loss", "register_loss"]
