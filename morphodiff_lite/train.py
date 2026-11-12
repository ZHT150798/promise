"""Adversarial distillation of a single-scale MorphoDiff-Lite student.

Adapted from `train.py` of AdcSR (Adversarial Diffusion Compression, CVPR 2025,
https://github.com/Guaishou74851/AdcSR, Apache-2.0). Changes made for PROMISE:

  * the teacher is a trained MorphoDiff checkpoint (LoRA on SD-Turbo) instead of OSEDiff;
  * the teacher is conditioned on one fixed scale-aware prompt instead of DAPE/RAM tags;
  * the discriminator is a copy of the MorphoDiff teacher U-Net (with its LoRA adapters)
    instead of a general-purpose Stable Diffusion U-Net with fresh LoRA adapters;
  * low-resolution inputs come from `RealESRGANDegrader.degrade_wsi_compress`
    (resize + JPEG) instead of the two-order Real-ESRGAN degradation;
  * single-process training with gradient accumulation instead of DDP.

Run from this directory:  python train.py --teacher_ckpt /path/to/morphodiff.pkl ...
"""
import torch, os, random, copy
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from argparse import ArgumentParser
from time import time
from tqdm import tqdm
from omegaconf import OmegaConf
from peft import LoraConfig

from dataset import RealESRGANDataset, RealESRGANDegrader
from model import Net
from build import build_half_decoder, load_half_decoder_weights

DEFAULT_PROMPT = ("20× magnification, 4× super-resolution, A high-resolution, Clear tissue borders, "
                  "Clear morphological features, high magnification.")


def parse_args(argv=None):
    parser = ArgumentParser()
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--model_dir", type=str, default="weight")
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulate steps")
    parser.add_argument("--config", type=str, default="config.yml",
                        help="Data/degradation config; set dataroot_gt and scale there")
    parser.add_argument("--sd_turbo_path", type=str, default="stabilityai/sd-turbo",
                        help="SD-Turbo base model (local directory or Hugging Face id)")
    parser.add_argument("--teacher_ckpt", type=str, required=True,
                        help="Trained MorphoDiff teacher checkpoint (.pkl)")
    parser.add_argument("--halfdecoder_ckpt", type=str, default="./weight/pretrained/halfDecoder.ckpt",
                        help="Pretrained channel-pruned VAE decoder released by AdcSR")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
                        help="Fixed teacher prompt of the (magnification, scale) pair being distilled")
    return parser.parse_args(argv)


def main(args):
    rank = 0

    # fixed seed for reproduction
    seed = rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    config = OmegaConf.load(args.config)

    epoch = args.epoch
    learning_rate = args.learning_rate
    bsz = args.batch_size
    accumulation_steps = args.accumulation_steps

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    print("gradient accumulation steps =", accumulation_steps)

    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(args.sd_turbo_path).to(device)
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    unet = pipe.unet
    text_encoder = pipe.text_encoder

    # ---- teacher: SD-Turbo + MorphoDiff LoRA weights ----
    vae_teacher = copy.deepcopy(vae)
    unet_teacher = copy.deepcopy(unet)
    osediff = torch.load(args.teacher_ckpt, weights_only=False)
    lora_conf_encoder = LoraConfig(r=osediff["rank_unet"], init_lora_weights="gaussian",
                                   target_modules=osediff["unet_lora_encoder_modules"])
    lora_conf_decoder = LoraConfig(r=osediff["rank_unet"], init_lora_weights="gaussian",
                                   target_modules=osediff["unet_lora_decoder_modules"])
    lora_conf_others = LoraConfig(r=osediff["rank_unet"], init_lora_weights="gaussian",
                                  target_modules=osediff["unet_lora_others_modules"])
    unet_teacher.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet_teacher.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet_teacher.add_adapter(lora_conf_others, adapter_name="default_others")
    for n, p in unet_teacher.named_parameters():
        if "lora" in n or "conv_in" in n:
            p.data.copy_(osediff["state_dict_unet"][n])
    unet_teacher.set_adapter(["default_encoder", "default_decoder", "default_others"])
    vae_lora_conf_encoder = LoraConfig(r=osediff["rank_vae"], init_lora_weights="gaussian",
                                       target_modules=osediff["vae_lora_encoder_modules"])
    vae_teacher.add_adapter(vae_lora_conf_encoder, adapter_name="default_encoder")
    for n, p in vae_teacher.named_parameters():
        if "lora" in n:
            p.data.copy_(osediff["state_dict_vae"][n])
    vae_teacher.set_adapter(["default_encoder"])
    del osediff

    # ---- discriminator: initialised from the MorphoDiff teacher U-Net ----
    unet_D = copy.deepcopy(unet_teacher)
    new_conv_in = torch.nn.Conv2d(256, 320, 3, padding=1).to(device)
    new_conv_in.weight.data = unet_D.conv_in.weight.data.repeat(1, 64, 1, 1) / 64
    new_conv_in.bias.data = unet_D.conv_in.bias.data
    unet_D.conv_in = new_conv_in

    # ---- channel-pruned VAE decoder (frozen) ----
    decoder = build_half_decoder().to(device)
    load_half_decoder_weights(decoder, args.halfdecoder_ckpt)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae_teacher.requires_grad_(False)
    unet_teacher.requires_grad_(False)
    decoder.requires_grad_(False)

    model = Net(unet, copy.deepcopy(decoder)).to(device)
    model_D = unet_D.to(device)

    model.requires_grad_(True)
    model_D.requires_grad_(False)
    params_to_opt = []
    for n, p in model_D.named_parameters():
        if "lora" in n or "conv_in" in n:
            p.requires_grad = True
            params_to_opt.append(p)

    param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("#Param.", param_cnt/1e6, "M")

    dataset = RealESRGANDataset(config, accumulation_steps*bsz)
    degrader = RealESRGANDegrader(config, device)
    dataloader = DataLoader(dataset, batch_size=bsz, num_workers=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-6)
    optimizer_D = torch.optim.Adam(params_to_opt, lr=1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epoch//2,], gamma=0.5)
    scaler = torch.cuda.amp.GradScaler()

    model_dir = "./%s" % (args.model_dir,)
    log_path = "./%s/log.txt" % (args.log_dir,)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print("start training...")
    timesteps = torch.tensor([999], device=device).long().expand(bsz,)
    alpha = pipe.scheduler.alphas_cumprod[999]
    del pipe
    for epoch_i in range(1, epoch + 1):
        start_time = time()
        loss_avg = 0.0
        loss_distil_avg = 0.0
        loss_adv_avg = 0.0
        loss_D_avg = 0.0
        iter_num = 0
        step_num = 0  # number of optimizer updates

        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch_i}")
        for batch_idx, batch in progress_bar:
            with torch.amp.autocast(device_type='cuda', enabled=True):
                with torch.no_grad():
                    LR, HR = degrader.degrade_wsi_compress(batch)
                    text_input = tokenizer([args.prompt],
                                           max_length=tokenizer.model_max_length,
                                           padding="max_length", return_tensors="pt").to(device)
                    encoder_hidden_states = text_encoder(text_input.input_ids.expand(bsz, -1), return_dict=False)[0]
                    LR, HR = LR * 2 - 1, HR * 2 - 1
                    LR_ = F.interpolate(LR, scale_factor=4, mode="bicubic")
                    LR_latents = vae_teacher.encode(LR_).latent_dist.mean * vae_teacher.config.scaling_factor
                    HR_latents = vae.encode(HR).latent_dist.mean
                    # one-step teacher prediction with the positive prompt only (no CFG)
                    pred_teacher = unet_teacher(
                        LR_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        return_dict=False,
                    )[0]
                    z0_teacher = (LR_latents-((1-alpha)**0.5)*pred_teacher)/(alpha**0.5)
                    z0_teacher = vae_teacher.post_quant_conv(z0_teacher / vae_teacher.config.scaling_factor)
                    z0_teacher = decoder.conv_in(z0_teacher)
                    z0_teacher = decoder.mid_block(z0_teacher)
                    z0_gt = vae.post_quant_conv(HR_latents)
                    z0_gt = decoder.conv_in(z0_gt)
                    z0_gt = decoder.mid_block(z0_gt)
                z0_student = model(LR)
                loss_distil = (z0_student - z0_teacher).abs().mean()
                loss_adv = F.softplus(-model_D(
                    z0_student,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )[0]).mean()
                loss = loss_distil + loss_adv
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            if (iter_num + 1) % accumulation_steps == 0 or (iter_num + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=True):
                pred_real = model_D(
                    z0_gt.detach(),
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )[0]
                pred_fake = model_D(
                    z0_student.detach(),
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )[0]
                loss_D = F.softplus(pred_fake).mean() + F.softplus(-pred_real).mean()
            loss_D = loss_D / accumulation_steps
            scaler.scale(loss_D).backward()
            if (iter_num + 1) % accumulation_steps == 0 or (iter_num + 1) == len(dataloader):
                scaler.step(optimizer_D)
                scaler.update()
                optimizer_D.zero_grad(set_to_none=True)
                step_num += 1

            loss_avg += loss.item() * accumulation_steps
            loss_distil_avg += loss_distil.item()
            loss_adv_avg += loss_adv.item()
            loss_D_avg += loss_D.item() * accumulation_steps
            iter_num += 1

            progress_bar.set_postfix({
                "loss": f"{loss_avg / iter_num:.4f}",
                "distil": f"{loss_distil_avg / iter_num:.4f}",
                "adv": f"{loss_adv_avg / iter_num:.4f}",
                "D": f"{loss_D_avg / iter_num:.4f}",
                "steps": step_num,
            })

        scheduler.step()
        loss_avg /= iter_num
        loss_distil_avg /= iter_num
        loss_adv_avg /= iter_num
        loss_D_avg /= iter_num

        log_data = "[%d/%d] Average loss: %f, distil loss: %f, adv loss: %f, D loss: %f, time cost: %.2fs, cur lr is %f." % (epoch_i, epoch, loss_avg, loss_distil_avg, loss_adv_avg, loss_D_avg, time() - start_time, scheduler.get_last_lr()[0])

        print(log_data)
        with open(log_path, "a") as log_file:
            log_file.write(log_data + "\n")
        if epoch_i % args.save_interval == 0:
            torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))


if __name__ == "__main__":
    main(parse_args())
