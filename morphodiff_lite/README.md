# MorphoDiff-Lite

MorphoDiff-Lite is the distilled student of MorphoDiff. It is obtained with
Adversarial Diffusion Compression (ADC), and the code in this directory is
modified from the official ADC implementation, AdcSR:

> Chen et al., *Adversarial Diffusion Compression for Real-World Image
> Super-Resolution*, CVPR 2025. <https://github.com/Guaishou74851/AdcSR>

AdcSR is released under the Apache License 2.0; a copy is kept in
[LICENSE](LICENSE).

## Environment

MorphoDiff-Lite requires newer versions of PyTorch, diffusers and peft than
MorphoDiff, so create a separate environment for it:

```bash
conda create -n morphodiff-lite python=3.10
```

```bash
conda activate morphodiff-lite
```

```bash
cd morphodiff_lite
```

```bash
pip install -r requirements.txt
```

Run the commands below from `morphodiff_lite/`.

## Weights

- SD-Turbo (`stabilityai/sd-turbo`).
- A trained MorphoDiff checkpoint (`.pkl`), used as the teacher during
  distillation.
- `halfDecoder.ckpt`, the channel-pruned VAE decoder released with AdcSR
  (<https://huggingface.co/Guaishou74851/AdcSR>). Place it in
  `./weight/pretrained/`.

## Distillation

Set `dataroot_gt` in `config.yml` to the training patch directory:

```text
dataroot_gt/
├── hr/   # 512×512 high-resolution PNG patches
└── lr/   # low-resolution counterparts, same file names
```

Then run:

```bash
python train.py \
  --sd_turbo_path /path/to/sd-turbo \
  --teacher_ckpt /path/to/morphodiff.pkl \
  --model_dir weight/20x_x4 \
  --log_dir log/20x_x4
```

One student is distilled for each magnification and upscaling factor. Pass the
teacher prompt of that setting with `--prompt`. The default is:

```text
20× magnification, 4× super-resolution, A high-resolution, Clear tissue borders, Clear morphological features, high magnification.
```

| Option | Default | Description |
| --- | --- | --- |
| `--epoch` | 200 | Number of epochs; the learning rate is halved halfway |
| `--learning_rate` | 1e-4 | Initial learning rate of the student |
| `--batch_size` | 1 | Patches per forward pass |
| `--accumulation_steps` | 8 | Gradient accumulation steps; effective batch size is `batch_size × accumulation_steps` |
| `--save_interval` | 10 | Save `<model_dir>/net_params_<epoch>.pkl` every N epochs |
| `--halfdecoder_ckpt` | `./weight/pretrained/halfDecoder.ckpt` | Pruned decoder weights |

## Inference

```bash
python infer.py \
  --input_image /path/to/lr_patches \
  --output_dir /path/to/sr_patches \
  --student_ckpt weight/20x_x4/net_params_200.pkl \
  --sd_turbo_path /path/to/sd-turbo \
  --upscale 4
```

`--input_image` is a 128×128 patch or a directory of patches; results are saved
as PNG in `--output_dir`. For `--upscale 8`, the input is upsampled ×2 with
bicubic interpolation before it is passed to the student.
