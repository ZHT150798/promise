# Setup and inference

Run commands from the repository root. Training and model inference require a
CUDA GPU. The dependency files pin PyTorch 2.0.1 and xformers 0.0.20; use a
compatible Python and CUDA environment.

## Dependencies

For the CUDA 11.7 build of PyTorch:

```bash
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
```

The training and evaluation scripts also import the following packages:

```bash
python -m pip install accelerate lpips pytorch-fid timm vision-aided-loss
```

Whole-slide processing additionally uses OpenSlide, libvips, and their Python
bindings, plus patchify. Install the native libraries for your operating system
before installing the bindings:

```bash
python -m pip install openslide-python pyvips patchify
```

## Model weights

| Model | Purpose | Configuration |
| --- | --- | --- |
| Stable Diffusion base model | Diffusion UNet, VAE, tokenizer, and scheduler | `paths.sd_model` for training; `--pretrained_model_name_or_path` for inference |
| PROMISE checkpoint | Trained LoRA adapters and convolution weights | `model.osediff_path` for training; `--osediff_path` for inference |
| CONCH v1.5 | Morphological consistency loss and pathology feature evaluation | `paths.conch_bin` for training; `build_conch()` in `my_utils/fd_loss/conch.py` for standalone scripts |
| UNI or Prov-GigaPath | Alternative pathology feature encoders | `loss.pe_type` and the corresponding checkpoint field in the training path configuration |

The training path examples use SD-turbo. A PROMISE checkpoint must be paired
with the diffusion base model used to train it. Set `model.osediff_path=null`
when training fresh LoRA adapters.

Standalone scripts that call `build_conch()` without an argument use its default
checkpoint path. Set that path for your installation before running patch
inference or patch extraction. The configuration-driven trainer passes the CONCH
path explicitly.

## Training

See [PROMISE training](../training/README.md) for paired-patch preparation,
configuration overrides, training stages, and tests.

## Patch reconstruction and evaluation

```bash
python run/infer/patch_level_infer.py \
  --input_image /path/to/lr_patches \
  --ref_path /path/to/hr_patches \
  --output_dir /path/to/reconstructed_patches \
  --pretrained_model_name_or_path /path/to/sd-turbo \
  --osediff_path /path/to/morphodiff.pkl \
  --upscale 4 \
  --align_method wavelet \
  --mixed_precision fp16
```

The script saves reconstructed PNGs and calculates FID, PSNR, LPIPS, and pathology
feature error against the reference directory. It currently sets
`CUDA_VISIBLE_DEVICES='1'` near the top of the file; adjust this to the GPU
available on your machine.

## Whole-slide reconstruction

```bash
python run/infer/wsi_infer_to_tif.py \
  --input_image /path/to/wsi_files \
  --output_dir /path/to/reconstructed_slides \
  --coord_dir /path/to/segmentation_coordinates \
  --pretrained_model_name_or_path /path/to/sd-turbo \
  --osediff_path /path/to/morphodiff.pkl \
  --mag 40x \
  --upscale 4 \
  --save_format tif
```

The script supports OpenSlide-readable slides and ordinary image inputs.
`--save_format tif` selects TIFF output. Match the magnification, scale, and text
conditioning to the checkpoint and input data.

## Patch extraction

`scripts/getPatch.py` extracts paired patches using CONCH-based sampling:

```bash
PYTHONPATH=. python scripts/getPatch.py \
  --path /path/to/wsi_files \
  --seg_path /path/to/segmentation_coordinates \
  --save_path /path/to/new_patch_directory \
  --patch_size 512 \
  --max_samples 10000
```

The script removes an existing output directory before writing patches; use a
new destination. It also contains installation-specific GPU and import-path
settings near the top of the file that should be adjusted locally.

## Model and checkpoint structure

`new_osediff.py` defines `OSEDiff_gen` for training, `OSEDiff_test` for inference,
and `OSEDiff_reg` for the optional regularization path. The model encodes the
low-resolution input with the VAE, applies a single denoising step, and decodes
the reconstructed image. Magnification and super-resolution scale are encoded
in text conditioning, for example `40× magnification, 4× super-resolution`.

Checkpoints store the UNet and VAE LoRA ranks, adapter module names, and trained
state dictionaries. They do not contain the complete diffusion base model.
The default training ranks are 32 for the UNet and 16 for the VAE.

Training images and model outputs use the range `[-1, 1]`; image saving converts
outputs to `[0, 1]`. The pathology feature losses apply their own normalization.
VAE and latent tiling are available through inference arguments for larger
images.
