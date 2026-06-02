import pytest

from promise_train.models.morphodiff import (
    CHECKPOINT_KEYS,
    PAPER_RANK_UNET,
    PAPER_RANK_VAE,
    validate_checkpoint_schema,
)


def _paper_rank_checkpoint() -> dict[str, object]:
    checkpoint = {key: object() for key in CHECKPOINT_KEYS}
    checkpoint["rank_unet"] = PAPER_RANK_UNET
    checkpoint["rank_vae"] = PAPER_RANK_VAE
    return checkpoint


def test_checkpoint_schema_keys() -> None:
    checkpoint = _paper_rank_checkpoint()

    validate_checkpoint_schema(checkpoint)
    assert checkpoint["rank_unet"] == 32
    assert checkpoint["rank_vae"] == 16


def test_checkpoint_schema_rejects_missing_key() -> None:
    checkpoint = _paper_rank_checkpoint()
    checkpoint.pop("rank_unet")

    with pytest.raises(KeyError):
        validate_checkpoint_schema(checkpoint)


def test_checkpoint_schema_rejects_invalid_rank() -> None:
    checkpoint = _paper_rank_checkpoint()
    checkpoint["rank_vae"] = 0

    with pytest.raises(ValueError):
        validate_checkpoint_schema(checkpoint)
