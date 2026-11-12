"""Shared constructors for MorphoDiff-Lite (PROMISE).

The channel-pruned VAE decoder definition and its checkpoint-loading logic are
taken unchanged from the AdcSR training/test scripts
(https://github.com/Guaishou74851/AdcSR, Apache-2.0); they are only factored
into functions here so that training and inference share them.
"""
import torch
from diffusers.models.autoencoders.vae import Decoder


def build_half_decoder():
    """Channel-pruned VAE decoder ("halfDecoder") used by AdcSR.

    The Stable Diffusion VAE decoder uses block_out_channels
    [128, 256, 512, 512]; this decoder uses [64, 128, 256, 256].
    """
    return Decoder(in_channels=4,
                   out_channels=3,
                   up_block_types=["UpDecoderBlock2D" for _ in range(4)],
                   block_out_channels=[64, 128, 256, 256],
                   layers_per_block=2,
                   norm_num_groups=32,
                   act_fn="silu",
                   norm_type="group",
                   mid_block_add_attention=True)


def load_half_decoder_weights(decoder, ckpt_path):
    """Load the pretrained pruned decoder released by AdcSR (halfDecoder.ckpt)."""
    ckpt_halfdecoder = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    decoder_ckpt = {}
    for k, v in ckpt_halfdecoder["state_dict"].items():
        if "decoder" in k:
            new_k = k.replace("decoder.", "")
            decoder_ckpt[new_k] = v
    decoder.load_state_dict(decoder_ckpt, strict=True)
    return decoder


def decoder_tail(decoder):
    """Layers appended to the student at inference to map mid-block features to pixels."""
    return [*decoder.up_blocks, decoder.conv_norm_out, decoder.conv_act, decoder.conv_out]
