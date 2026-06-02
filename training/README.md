# PROMISE training

The `promise_train` package provides configuration-driven training with Hydra,
single- and multi-GPU execution through Accelerate, and patch-level validation.
Run the commands below from the repository root in an environment with the
project dependencies installed.

## Model weights and data

Set the paths in `training/conf/paths/local.yaml`, or override them on the command
line. Training requires a Stable Diffusion base model and the CONCH v1.5 weights.
Use the same diffusion base model for training and inference. Set
`model.osediff_path=null` to initialize fresh LoRA adapters, or provide a PROMISE
checkpoint to load trained weights.

For the default `wsisr_compress` dataset, `data.train_path` and `data.val_path`
are directories containing paired patches with matching filenames:

```text
dataset/
├── train/
│   ├── hr/
│   └── lr/
└── val/
    ├── hr/
    ├── lr/
    └── lr_X8/   # needed only when evaluating scale 8
```

Training reads PNG pairs. Validation reads PNG reference patches and accepts PNG
or JPEG low-resolution patches. Supported validation patch sizes are 512 and
1024 pixels. Configure the magnification and scale pairs with `data.val_specs`.

## Train

```bash
PYTHONPATH=.:training/src accelerate launch \
  --config_file training/conf/acc/1gpu.yaml \
  -m promise_train.entrypoints.train \
  paths=local \
  paths.sd_model=/path/to/sd-turbo \
  paths.conch_bin=/path/to/conch_v1_5_pytorch_model.bin \
  model.osediff_path=null \
  data.train_path=/path/to/dataset/train \
  data.val_path=/path/to/dataset/val \
  output_dir=experience/promise
```

For four GPUs, replace `training/conf/acc/1gpu.yaml` with
`training/conf/acc/4gpu.yaml`. The `paths=cluster` profile provides a separate set
of paths for cluster installations; fill in the paths before using it.

The entrypoint saves `resolved_config.yaml` and `environment.txt` in the output
directory. Hydra also records each run's configuration under `outputs/` within
that directory. Checkpoints, evaluation outputs, and logs are excluded from Git.

## Training stages

The default schedule uses reconstruction losses first, then enables adversarial
training:

```yaml
train:
  max_steps: 150000
  lr: 5e-5
  lr_stage2: 2e-5
  gan_start_step: 100000
```

At `gan_start_step`, training saves a `stage1_final` checkpoint, switches the
learning rate, resets the constant learning-rate schedulers, and enables the GAN
updates. Set `gan_start_step=0` for adversarial training from the start, or
`gan_start_step=null` to disable that stage.

Default loss weights are L2 = 2, LPIPS = 4, morphological consistency = 1, and
GAN = 0.05. The default LoRA ranks are 32 for the UNet and 16 for the VAE.

## Tests

```bash
PYTHONPATH=.:training/src pytest -q training/tests
```

The suite checks numerical equivalence of degradation operations, reconstruction
updates and training trajectories; training-stage transitions; evaluation and
distributed aggregation; checkpoint structure; and the training entrypoint.
Distributed tests launch two CPU processes and require a local rendezvous port.

These tests use small synthetic inputs and test doubles where appropriate.
They do not replace evaluation on pathology data.

## Check the training entrypoint

This one-step run uses synthetic data and a small test model, without downloading
pretrained weights:

```bash
PYTHONPATH=.:training/src python -m promise_train.entrypoints.train \
  model=stub loss=stub data=stub output_dir=experience/entrypoint_check \
  train.max_steps=1 train.num_epochs=1 train.batch_size=1 \
  train.grad_accum=1 train.dataloader_num_workers=0 train.ckpt_steps=99 \
  train.gan_start_step=null train.mixed_precision=no train.report_to=null \
  train.val.skip_fid=true
```

The project runtime dependencies are still required for imports.

## Inspect learning curves

`training/scripts/sanity_train_curve.py` records training losses and checks
learning-curve behavior. It uses the same Hydra configuration as training.
For a small-subset overfitting check with fresh LoRA adapters:

```bash
PYTHONPATH=.:training/src python training/scripts/sanity_train_curve.py \
  paths.sd_model=/path/to/sd-turbo \
  paths.conch_bin=/path/to/conch_v1_5_pytorch_model.bin \
  data.train_path=/path/to/dataset/train \
  data.val_path=/path/to/dataset/val \
  output_dir=experience/overfit_check \
  data.train_max_samples=16 model.osediff_path=null \
  loss.gan=false train.gan_start_step=null \
  train.max_steps=500 train.num_epochs=100 train.batch_size=1 train.grad_accum=1 \
  train.dataloader_num_workers=0 train.ckpt_steps=9999 \
  train.mixed_precision=fp16 train.val.skip_fid=true train.report_to=null \
  +sanity.mode=overfit +sanity.window=50 +sanity.print_every=10
```

For a trained checkpoint, use `+sanity.mode=stability`, set
`model.osediff_path=/path/to/morphodiff.pkl`, and enable GAN with
`loss.gan=true train.gan_start_step=150 train.max_steps=300`. Stability checks
require finite losses and bounded GAN losses; overfitting checks require a
decreasing reconstruction-loss trend. These checks require model weights, data,
and a CUDA GPU. Their results describe training behavior, not reconstruction
quality on a held-out dataset.

## Shared implementations

The package uses the model in `new_osediff.py`, degradation operations in
`dataloaders/realesrgan.py`, and pathology feature losses in `my_utils/fd_loss/`.
`run/train/train_multi_mag-scale.py` provides a standalone training entrypoint.
Equivalence tests compare the package against the shared operations and a
reference reconstruction-update implementation.
