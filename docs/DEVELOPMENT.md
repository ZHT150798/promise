# Development guide

Environment setup, training and inference commands, architecture, and data conventions for PROMISE.

---

## 项目简介

**PROMISE**（Pathology Reconstruction On-demand via Morphologically Informed Super-resolution Engine）是一个面向全切片图像（WSI）存档的超分辨率压缩框架，核心范式为"低分辨率存储、高分辨率按需重建"。核心模型 **MorphoDiff** 基于单步扩散（OSEDiff），通过 LoRA 适配器微调 Stable Diffusion 的 UNet 与 VAE，并引入以病理基础模型（CONCH v1.5）特征空间计算的 Morphological Consistency (MC) 损失来抑制幻觉。

---

## 环境安装

依赖版本锁定较严，建议严格遵守：

```bash
# 推荐 Python 3.9，CUDA 11.7
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
pip install xformers==0.0.20  # 必须与 torch 2.0.1 配套
pip install -r requirements.txt
```

额外依赖（requirements.txt 未列出，按需安装）：

```bash
pip install openslide-python pyvips patchify pytorch-fid vision-aided-loss
```

---

## 必要的模型权重路径

运行前需手动下载并配置以下权重：

| 模型 | 用途 | 配置方式 |
|------|------|----------|
| Stable Diffusion v2.1（或自定义 SD 模型） | UNet / VAE / Tokenizer 基座 | `--pretrained_model_name_or_path` 参数 |
| CONCH v1.5 (`conch_v1_5_pytorch_model.bin`) | MC 损失 / PE 评估指标 | **硬编码**在 `my_utils/fd_loss/conch.py:688`，需直接修改该路径 |
| UNI (`pytorch_model.bin`) | 可选的 MC 损失替代（`--pe_type uni`） | `train_multi_mag-scale.py` 中硬编码路径 |
| Prov-GigaPath (`pytorch_model.bin`) | 可选的 MC 损失替代（`--pe_type gigapath`） | 同上 |

---

## 主要命令

### 训练

```bash
# 多放大倍率 + 多 scale 联合训练（主训练脚本）
accelerate launch run/train/train_multi_mag-scale.py \
  --pretrained_model_name_or_path /path/to/sd-model \
  --train_path /path/to/train_filelist.txt \
  --val_path /path/to/val_patches/ \
  --output_dir experience/wsisr_main \
  --pe_type conch \
  --lambda_pe 1.0 \
  --lambda_lpips 1.0 \
  --lambda_l2 1.0 \
  --lora_rank_unet 4 \
  --lora_rank_vae 4 \
  --train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --mixed_precision fp16 \
  --enable_xformers_memory_efficient_attention \
  --gan
```

训练数据通过 TXT 文件指定（每行一个高分辨率 patch 绝对路径），运行时在线生成降质 LR。

### Patch 级推理（含评估）

```bash
python run/infer/patch_level_infer.py \
  --input_image /path/to/lr_patches/ \
  --output_dir /path/to/output/ \
  --ref_path /path/to/hr_patches/ \
  --pretrained_model_name_or_path /path/to/sd-model \
  --osediff_path /path/to/morphodiff.pkl \
  --upscale 4 \
  --align_method wavelet \
  --mixed_precision fp16 \
  --use_cfg True
```

推理结束后自动计算并输出 FID / PSNR / LPIPS / PE，保存至 `output_dir/metrics.csv`。

### WSI 级推理（输出 TIF）

```bash
python run/infer/wsi_infer_to_tif.py \
  --input_image /path/to/wsi_files/ \
  --output_dir /path/to/output_tif/ \
  --coord_dir /path/to/seg_coords/ \
  --pretrained_model_name_or_path /path/to/sd-model \
  --osediff_path /path/to/morphodiff.pkl \
  --mag 40x
```

### 数据集准备（WSI → Patch）

```bash
# 第一步：从 WSI 中提取 HR/LR 配对 patch（基于 CONCH 特征的聚类采样）
python scripts/getPatch.py \
  --path /path/to/wsi_folder/ \
  --seg_path /path/to/seg_results/ \
  --save_path /path/to/output_patches/ \
  --patch_size 512 \
  --max_samples 10000
```

---

## 架构设计

### 模型类层次（`new_osediff.py`）

| 类名 | 用途 |
|------|------|
| `OSEDiff_gen` | 训练阶段生成器，含 LoRA 初始化逻辑 |
| `OSEDiff_test` | 推理阶段，支持 Tiled Latent（大 patch 分块处理）和 CFG |
| `OSEDiff_reg` | 分布匹配正则化辅助模块（训练时可选） |

三个类共享相同的 `load_ckpt()` 结构：从 `.pkl` 文件恢复 UNet（encoder/decoder/others 三套 LoRA）和 VAE（encoder LoRA）权重。

### 前向传播数据流

```
LR patch [-1,1]
  │
  ├─ VAE encoder (+ LoRA) → latent z
  │
  ├─ Text prompt (magnification + scale 文本) → CLIP → prompt_embeds
  │
  ├─ UNet (+ 三套 LoRA, t=999) → noise_pred
  │
  ├─ DDPMScheduler.step(1步) → x_denoised
  │
  └─ VAE decoder → SR patch [-1,1]
```

单步推理：固定使用 `timesteps = [999]`，`noise_scheduler.set_timesteps(1)`。

### MC 损失（Morphological Consistency Loss）

位于 `my_utils/fd_loss/pe_loss.py`，`PELoss` 类：

1. 将 SR 和 HR 图像从 `[-1,1]` 转换为 `[0,1]`
2. 对 40× patch（< 1024px）做 reflect padding 至 1024×1024
3. 经 `eval_transform`（Resize 448, CenterCrop 448, Normalize）后输入 CONCH v1.5
4. 计算 SR 与 HR 特征向量的 L1 距离作为损失

训练时总损失 = `λ_l2 * L2 + λ_lpips * LPIPS + λ_pe * MC + λ_gan * GAN`

### Scale-Aware 文本提示格式

文本提示编码了光学放大倍率和超分 scale，格式示例：

- `"40× magnification, 4× super-resolution"` → 40× WSI，4倍上采样
- `"20× magnification, 4× super-resolution"` → 20× WSI，4倍上采样

正向默认提示：`"A high-resolution, Clear tissue borders, Clear morphological features, high magnification."`  
负向默认提示：`"blur, low quality, low resolution, oversmooth, low magnification."`

在推理时通过 `--pos_prompt` 和 `--neg_prompt` 传入，训练时同样支持。

### Tiled 推理（内存管理）

`OSEDiff_test` 自动处理大 patch：

- **VAE Tiled Hook**（`my_utils/vaehook.py`）：encoder tile size 默认 1024，decoder 默认 224
- **Latent Tiled Inference**：latent tile size 默认 96，overlap 默认 32，使用 Gaussian 权重融合重叠区域
- 两级 tiling 合并保证任意尺寸输入均可推理

### 检查点格式（`.pkl`）

`model.save_model()` 保存字段：

```python
{
  "rank_unet": int,
  "rank_vae": int,
  "unet_lora_encoder_modules": List[str],
  "unet_lora_decoder_modules": List[str],
  "unet_lora_others_modules": List[str],
  "vae_lora_encoder_modules": List[str],
  "state_dict_unet": {LoRA 和 conv_in 权重},
  "state_dict_vae": {LoRA 权重},
}
```

只保存 LoRA delta 权重和 `unet.conv_in`（6通道输入层），不保存 SD 基础权重。

---

## 数据集结构约定

### 训练数据 TXT 清单

TXT 文件每行一个 **HR patch 绝对路径**（.jpg 或 .png），LR 在线通过 `RealESRGAN_degradation` 动态生成。

### WSI 数据集目录结构（推理）

```
dataset/
├── input/      # 低分辨率 WSI 文件 (.svs, .tiff 等)
├── seg/        # 前景分割坐标（由 wsi_utils.segment_foreground_background 生成）
└── gt/         # 原始高分辨率 WSI（仅评估时需要）
```

### 癌症类型文本提示映射

`dataloaders/wsisr_dataset.py` 中的 `cancer_infor` 字典定义了 11 种癌症类型（GBM、HCC、LUAD、BRCA 等）对应的自然语言描述，可用于增强文本条件。

---

## 图像值域约定

| 阶段 | 值域 |
|------|------|
| 数据集输出 | `[-1, 1]`（经 `Normalize(mean=0.5, std=0.5)` 处理） |
| 模型输入/输出 | `[-1, 1]` |
| 保存/可视化前 | `output * 0.5 + 0.5` 还原至 `[0, 1]` |
| CONCH 输入 | `[0, 1]` → ImageNet Normalize |

---

## 评估指标

`my_utils/metrics.py` 提供三个函数，接受两个目录路径作为输入：

- `psnr(dir1, dir2)` — 像素级保真度
- `lpips_score(dir1, dir2, lpips_loss, device)` — 感知质量
- `pe_score(dir1, dir2, pe_loss_fn, device)` — 病理语义一致性（CONCH 特征 L1）

`patch_level_infer.py` 推理结束后自动调用以上三项，并额外计算 FID。

---

## 已知注意事项

- `my_utils/fd_loss/conch.py:688` 中 CONCH 权重路径为**硬编码绝对路径**，切换机器时必须修改此行。
- 同样，UNI 和 Prov-GigaPath 权重路径也硬编码在 `train_multi_mag-scale.py` 的 `main()` 函数中，`--pe_type uni/gigapath` 前需先修改。
- `run/infer/patch_level_infer.py` 顶部硬编码了 `os.environ['CUDA_VISIBLE_DEVICES'] = '1'`，多卡环境下需注意。
- 训练脚本通过 `accelerate launch` 启动，多 GPU 配置通过 `accelerate config` 预设。
