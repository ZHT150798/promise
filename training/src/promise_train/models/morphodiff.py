from typing import Any

CHECKPOINT_KEYS = {
    "rank_unet",
    "rank_vae",
    "unet_lora_encoder_modules",
    "unet_lora_decoder_modules",
    "unet_lora_others_modules",
    "vae_lora_encoder_modules",
    "state_dict_unet",
    "state_dict_vae",
}

PAPER_RANK_UNET = 32
PAPER_RANK_VAE = 16


def build_morphodiff_gen(args: Any):
    from new_osediff import OSEDiff_gen

    return OSEDiff_gen(args)


def build_morphodiff_test(args: Any):
    from new_osediff import OSEDiff_test

    return OSEDiff_test(args)


def validate_checkpoint_schema(checkpoint: dict[str, Any]) -> None:
    missing = CHECKPOINT_KEYS.difference(checkpoint)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"Missing MorphoDiff checkpoint keys: {missing_str}")
    for key in ("rank_unet", "rank_vae"):
        value = checkpoint[key]
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Invalid MorphoDiff checkpoint rank {key}: {value}")


__all__ = [
    "CHECKPOINT_KEYS",
    "PAPER_RANK_UNET",
    "PAPER_RANK_VAE",
    "build_morphodiff_gen",
    "build_morphodiff_test",
    "validate_checkpoint_schema",
]
