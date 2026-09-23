# PROMISE

Official implementation of **PROMISE** (Pathology Reconstruction On-demand via
Morphologically Informed Super-resolution Engine), a framework for storing
whole-slide pathology images at low resolution and reconstructing them on demand.

The MorphoDiff model uses single-step diffusion with LoRA adapters and a
morphological consistency loss based on pathology image features.

## Getting started

1. Follow the [setup and inference guide](docs/DEVELOPMENT.md) to install the
   dependencies and configure model weights.
2. See the [training guide](training/README.md) for the dataset layout,
   configuration, single- and multi-GPU training, and tests.
3. Use `run/infer/patch_level_infer.py` for patch reconstruction and evaluation,
   or `run/infer/wsi_infer_to_tif.py` for whole-slide reconstruction.

## Repository structure

| Path | Contents |
| --- | --- |
| `new_osediff.py` | MorphoDiff training and inference models |
| `models/` | Diffusion UNet and VAE implementations |
| `training/` | Configurations, trainer, evaluator, and tests |
| `dataloaders/` | Dataset and image degradation implementations |
| `run/` | Standalone training, inference, and evaluation scripts |
| `my_utils/` | Pathology feature losses, image metrics, and WSI utilities |
| `scripts/` | Patch extraction and sampling tools |

Model weights and pathology datasets are not included in this repository.
Configure their locations before running training or inference.
