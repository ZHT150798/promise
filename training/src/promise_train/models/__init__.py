from collections.abc import Callable
from typing import Any

from .morphodiff import CHECKPOINT_KEYS, build_morphodiff_gen, build_morphodiff_test
from .stub import build_stub_morphodiff_gen

ModelBuilder = Callable[..., Any]
_MODELS: dict[str, ModelBuilder] = {}


def register_model(name: str, builder: ModelBuilder) -> None:
    if name in _MODELS:
        raise ValueError(f"Model already registered: {name}")
    _MODELS[name] = builder


def build_model(name: str, **kwargs: Any) -> Any:
    try:
        builder = _MODELS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown model: {name}") from exc
    return builder(**kwargs)


register_model("morphodiff_gen", build_morphodiff_gen)
register_model("morphodiff_test", build_morphodiff_test)
register_model("stub_gen", build_stub_morphodiff_gen)

__all__ = [
    "CHECKPOINT_KEYS",
    "build_model",
    "build_morphodiff_gen",
    "build_morphodiff_test",
    "build_stub_morphodiff_gen",
    "register_model",
]
