import sys
import os

# Standalone training entrypoint; also serves as a numerical reference in tests.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


import lpips
import argparse

# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import torch
torch.set_num_threads(1)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import SequentialSampler
import diffusers
import csv
import gc
import timm
import numpy as np
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from new_osediff import OSEDiff_gen
from dataloaders.wsisr_dataset import WSISRDataset, WSISRDataset_val_multi

from pathlib import Path
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs
from my_utils.fd_loss.conch import build_conch
from my_utils.fd_loss import pe_loss
from torch.utils.data.distributed import DistributedSampler
import tempfile
import shutil
from pathlib import Path
from pytorch_fid.fid_score import calculate_fid_given_paths


def parse_float_list(arg):
    try:
        return [float(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List elements should be floats")


def parse_int_list(arg):
    try:
        return [int(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List elements should be integers")


def parse_str_list(arg):
    return arg.split(',')


def parse_args(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()

    parser.add_argument("--revision", type=str, default=None, )
    parser.add_argument("--variant", type=str, default=None, )
    parser.add_argument("--tokenizer_name", type=str, default=None)

    # training details
    parser.add_argument("--gan_disc_type", default="vagan")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--output_dir", default='experience/osepath')
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512, )
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--num_samples_eval", type=int, default=500, help="Number of samples to use for all evaluation")
    parser.add_argument("--max_train_steps", type=int, default=100000, )
    parser.add_argument("--checkpointing_steps", type=int, default=500, )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--gradient_checkpointing", action="store_true", )
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help=(
                            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                            ' "constant", "constant_with_warmup"]'
                        ),
                        )
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
                        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
                        )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=8, )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
                        help=(
                            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
                            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
                        ),
                        )
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        help=(
                            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
                        ),
                        )
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], )
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true",
                        help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true", )
    parser.add_argument("--logging_dir", type=str, default="logs")

    parser.add_argument("--tracker_project_name", type=str, default="train_osediff",
                        help="The name of the wandb project to log to.")
    parser.add_argument('--train_path', type=str, default='YOUR TXT FILE PATH',
                        help='A comma-separated list of integers')
    parser.add_argument('--val_path', type=str,
                        default='/nas/zhanghongtai/SR/WSI_SR_Protocol/wsisr_pipeline/preset/datasets/tcga_l/val/patch_1024/',
                        help='A comma-separated list of integers')
    parser.add_argument("--deg_type", default="wsisr_compress", type=str,
                        choices=["norm", "realgan", "lr", "compress, wsisr_compress"])
    parser.add_argument("--deg_file_path", default="params_realesrgan.yml", type=str)
    parser.add_argument("--pretrained_model_name_or_path", default=None, type=str)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_lpips", default=1.0, type=float)
    parser.add_argument("--lambda_pe", default=1.0, type=float)
    parser.add_argument("--pos_prompt", type=str,
                        default="A high-resolution, Clear tissue borders, Clear morphological features, high magnification.")
    parser.add_argument("--neg_prompt", type=str,
                        default="blur, low quality,  low resolution, oversmooth, low magnification.")

    # lora setting
    parser.add_argument("--lora_rank_unet", default=4, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)
    parser.add_argument("--neg_prob", default=0.05, type=float)

    parser.add_argument("--gan", action="store_true", help="use_gan")
    parser.add_argument("--pe_type", default="conch", type=str, choices=["conch", "gigapath", "uni"])
    parser.add_argument("--osediff_path", type=str, default="/nas/zhanghongtai/SR/WSI_SR_Protocol/wsisr_pipeline/experience_multi/wsisr_multi-magscale_gan-compress/checkpoints/model_4001.pkl")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    model_gen = OSEDiff_gen(args)
    model_gen.set_train()
    total_params = sum(p.numel() for p in model_gen.parameters())
    trainable_params = sum(p.numel() for p in model_gen.parameters() if p.requires_grad)
    if accelerator.is_main_process:
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Pe_type: {args.pe_type}")

    if args.gan_disc_type == "vagan":
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(cv_type='dino', output_type='conv_multi_level',
                                                   loss_type=args.gan_loss_type, device="cuda")
    else:
        raise NotImplementedError(f"Discriminator type {args.gan_disc_type} not implemented")

    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    pe_type = args.pe_type
    if pe_type == "conch":
        conch, _ = build_conch()
        pe_loss_fn = pe_loss.PELoss(conch.cuda())
    elif pe_type == "uni":
        model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0,
                                  dynamic_img_size=True)
        uni_checkpoint = getattr(
            args,
            "uni_checkpoint",
            "/media/dell/data/zhangv1/WSISR/huggingface/UNI/pytorch_model.bin",
        )
        model.load_state_dict(
            torch.load(uni_checkpoint, map_location="cpu"),
            strict=True)
        pe_loss_fn = pe_loss.PELoss_uni(model.cuda())
    elif pe_type == "gigapath":
        gigapath_checkpoint = getattr(
            args,
            "gigapath_checkpoint",
            "/media/dell/data/zhangv1/WSISR/huggingface/prov-gigapath/pytorch_model.bin",
        )
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False,
                                  checkpoint_path=gigapath_checkpoint)
        pe_loss_fn = pe_loss.PELoss_prov(model.cuda())

    # set vae adapter
    model_gen.vae.set_adapter(['default_encoder'])
    # set gen adapter
    model_gen.unet.set_adapter(['default_encoder', 'default_decoder', 'default_others'])

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            model_gen.unet.enable_xformers_memory_efficient_attention()
            # model_reg.unet_fix.enable_xformers_memory_efficient_attention()
            # model_reg.unet_update.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        model_gen.unet.enable_gradient_checkpointing()
        # model_reg.unet_fix.enable_gradient_checkpointing()
        # model_reg.unet_update.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # make the optimizer
    layers_to_opt = []
    for n, _p in model_gen.unet.named_parameters():
        if "lora" in n:
            layers_to_opt.append(_p)
    layers_to_opt += list(model_gen.unet.conv_in.parameters())
    for n, _p in model_gen.vae.named_parameters():
        if "lora" in n:
            layers_to_opt.append(_p)

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                  eps=args.adam_epsilon, )
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                                 num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                 num_training_steps=args.max_train_steps * accelerator.num_processes,
                                 num_cycles=args.lr_num_cycles, power=args.lr_power, )
    optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=args.learning_rate,
                                       betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                       eps=args.adam_epsilon, )
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
                                      num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                      num_training_steps=args.max_train_steps * accelerator.num_processes,
                                      num_cycles=args.lr_num_cycles, power=args.lr_power)

    layers_to_opt_reg = []
    # for n, _p in model_reg.unet_update.named_parameters():
    #     if "lora" in n:
    #         layers_to_opt_reg.append(_p)
    # optimizer_reg = torch.optim.AdamW(layers_to_opt_reg, lr=args.learning_rate,
    #     betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
    #     eps=args.adam_epsilon,)
    # lr_scheduler_reg = get_scheduler(args.lr_scheduler, optimizer=optimizer_reg,
    #         num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
    #         num_training_steps=args.max_train_steps * accelerator.num_processes,
    #         num_cycles=args.lr_num_cycles, power=args.lr_power)

    dataset_train = WSISRDataset(split="train", args=args)
    dataset_val = WSISRDataset_val_multi(split="test", args=args)
    dataset_val_20x = WSISRDataset_val_multi(split="test", args=args, mag='20x')
    dataset_val_x8 = WSISRDataset_val_multi(split="test", args=args, scale=8)
    dataset_val_x8_20x = WSISRDataset_val_multi(split="test", args=args, mag='20x', scale=8)
    #dataset_val_x16 = WSISRDataset_val_compress(split="test", args=args, scale=16)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True,
                                           num_workers=args.dataloader_num_workers)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=8)
    dl_val_20x = torch.utils.data.DataLoader(dataset_val_20x, batch_size=1, shuffle=False, num_workers=8)
    dl_val_x8 = torch.utils.data.DataLoader(dataset_val_x8, batch_size=1, shuffle=False, num_workers=8)
    dl_val_x8_20x = torch.utils.data.DataLoader(dataset_val_x8_20x, batch_size=1, shuffle=False, num_workers=8)
    #dl_val, dl_val_20x, dl_val_x8, dl_val_x8_20x = accelerator.prepare(dl_val, dl_val_20x, dl_val_x8, dl_val_x8_20x)
    # 多卡val
    # dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, sampler=SequentialSampler(dataset_val), num_workers=8)
    # dl_val_20x = torch.utils.data.DataLoader(dataset_val_20x, batch_size=1, sampler=SequentialSampler(dataset_val_20x), num_workers=8)
    # dl_val_x8 = torch.utils.data.DataLoader(dataset_val_x8, batch_size=1, sampler=SequentialSampler(dataset_val_x8), num_workers=8)
    # dl_val_x8_20x = torch.utils.data.DataLoader(dataset_val_x8_20x, batch_size=1, sampler=SequentialSampler(dataset_val_x8_20x), num_workers=8)
    #dl_val_x16 = torch.utils.data.DataLoader(dataset_val_x16, batch_size=1, shuffle=False, num_workers=0)
    # 1) 构建原始 dataloader（不变）
    dl_val        = torch.utils.data.DataLoader(dataset_val,        batch_size=1, shuffle=False, num_workers=8)
    dl_val_20x    = torch.utils.data.DataLoader(dataset_val_20x,    batch_size=1, shuffle=False, num_workers=8)
    dl_val_x8     = torch.utils.data.DataLoader(dataset_val_x8,     batch_size=1, shuffle=False, num_workers=8)
    dl_val_x8_20x = torch.utils.data.DataLoader(dataset_val_x8_20x, batch_size=1, shuffle=False, num_workers=8)

    # 2) prepare（多卡切分，关闭重复样本填充）
    def make_val_loader(dataset, accelerator):
        sampler = DistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=False,
            drop_last=False,   # 不丢弃尾部样本，每张卡可能样本数差 1
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, sampler=sampler, num_workers=8
        )

    dl_val        = make_val_loader(dataset_val,        accelerator)
    dl_val_20x    = make_val_loader(dataset_val_20x,    accelerator)
    dl_val_x8     = make_val_loader(dataset_val_x8,     accelerator)
    dl_val_x8_20x = make_val_loader(dataset_val_x8_20x, accelerator)

    # 3) 组装 dl_list（不变）
    dl_list = [dl_val, dl_val_20x]  # , dl_val_x8, dl_val_x8_20x , dl_val_x8, dl_val_x8_20x

    # 4) 主 prepare（不包含 dl_val 系列）
    model_gen, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
        model_gen, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc
    )
    net_lpips  = accelerator.prepare(net_lpips)
    pe_loss_fn = accelerator.prepare(pe_loss_fn)
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False
    # renorm with image net statistics
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        args.train_path = str(args.train_path)
        args.val_path = str(args.val_path)
        #args.dataset_prob_paths_list = str(args.dataset_prob_paths_list)
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
                        disable=not accelerator.is_main_process, )

    # def evaluate_val_set(model, dl_list, net_lpips, pe_loss_fn, args, global_step, best_lpips, logs, save_vis=False):
    #     model.eval()
    #     device = next(model.parameters()).device

    #     # scales = ['40x', '40x_X8', '20x', '20x_X8']
    #     # val_results = {'40x': [], '40x_X8': [], '20x': [], '20x_X8': []}
    #     scales = ['40x', '20x']
    #     val_results = {'40x': [], '20x': []}
    #     all_lpips, all_pe, all_fid = [], [], []

    #     for scale, dl in zip(scales, dl_list):
    #         l_lpips, l_pe = [], []

    #         gt_dir = Path(args.output_dir, "temp", "gt")
    #         sr_dir = Path(args.output_dir, "temp", "sr")
    #         gt_dir.mkdir(parents=True, exist_ok=True)
    #         sr_dir.mkdir(parents=True, exist_ok=True)

    #         val_count = 0
    #         for step, batch in enumerate(dl):
    #             x_src = batch["conditioning_pixel_values"].to(device)
    #             x_tgt = batch["output_pixel_values"].to(device)
    #             B, C, H, W = x_src.shape
    #             assert B == 1, "Batch size must be 1 during eval."

    #             with torch.no_grad():
    #                 # 生成 prompt
    #                 batch["prompt"] = [batch["pos_prompt"][0] + ", " + args.pos_prompt for _ in range(B)]
    #                 x_tgt_pred, _, _ = model(x_src, batch=batch, prompt=batch["prompt"], args=args)

    #                 # LPIPS
    #                 loss_lpips = net_lpips(x_tgt_pred, x_tgt).mean()
    #                 l_lpips.append(loss_lpips.item())
    #                 all_lpips.append(loss_lpips.item())

    #                 # PE
    #                 if scale == '20x' or scale == '20x_X8':
    #                     loss_pe = pe_loss_fn(x_tgt_pred, x_tgt, mag_prompt="20× magnification")
    #                 else:
    #                     loss_pe = pe_loss_fn(x_tgt_pred, x_tgt)
    #                 l_pe.append(loss_pe.item())
    #                 all_pe.append(loss_pe.item())

    #                 # 保存用于 FID
    #                 for i in range(B):
    #                     gt_img_path = gt_dir / f"{step}_{i}.png"
    #                     sr_img_path = sr_dir / f"{step}_{i}.png"
    #                     transforms.ToPILImage()(x_tgt[i].cpu().clamp(0, 1)).save(gt_img_path)
    #                     transforms.ToPILImage()(x_tgt_pred[i].cpu().clamp(0, 1)).save(sr_img_path)

    #                 # 可视化前几个样本
    #                 if save_vis and val_count < 5:
    #                     to_vis = lambda x: x.cpu().detach() * 0.5 + 0.5
    #                     combined = torch.cat([to_vis(x_src), to_vis(x_tgt_pred), to_vis(torch.clamp(x_tgt, -1, 1))], dim=3)
    #                     output_pil = transforms.ToPILImage()(combined[0])
    #                     outf = os.path.join(args.output_dir, f"eval/val_{scale}_{step}.png")
    #                     output_pil.save(outf)
    #                     val_count += 1

    #         # 计算 scale 下 FID (PyTorch-FID)
    #         fid_value = calculate_fid_given_paths(
    #             [str(gt_dir), str(sr_dir)],
    #             batch_size=32,
    #             device=str(device),
    #             dims=2048
    #         )
    #         all_fid.append(fid_value)

    #         shutil.rmtree(gt_dir)
    #         shutil.rmtree(sr_dir)

    #         val_results[scale] = {
    #             "lpips": np.mean(l_lpips),
    #             "pe": np.mean(l_pe),
    #             "fid": fid_value
    #         }

    #     # === 平均结果 + 打印 ===
    #     avg_lpips = np.mean(all_lpips)
    #     avg_pe = np.mean(all_pe)
    #     avg_fid = np.mean(all_fid)

    #     logs["val/avg_lpips"] = avg_lpips
    #     logs["val/avg_pe"] = avg_pe
    #     logs["val/avg_fid"] = avg_fid
    #     for scale in scales:
    #         r = val_results[scale]
    #         logs[f"val/{scale}_lpips"] = r["lpips"]
    #         logs[f"val/{scale}_pe"] = r["pe"]
    #         logs[f"val/{scale}_fid"] = r["fid"]

    #     print(f"[Step {global_step}] Avg: LPIPS={avg_lpips:.4f}, PE={avg_pe:.4f}, FID={avg_fid:.4f}")
    #     for scale in scales:
    #         r = val_results[scale]
    #         print(f"  [{scale.upper()}] LPIPS={r['lpips']:.4f}, PE={r['pe']:.4f}, FID={r['fid']:.4f}")

    #     # === 保存最佳模型 ===
    #     if avg_lpips < best_lpips:
    #         best_lpips = avg_lpips
    #         outm = os.path.join(args.output_dir, "checkpoints",
    #                             f"bestmodel_{global_step}_{round(avg_lpips, 4)}_{round(avg_pe, 4)}.pkl")
    #         accelerator.unwrap_model(model).save_model(outm)

    #     # === 写 CSV ===
    #     log_file = os.path.join(args.output_dir, "eval/val_log.csv")
    #     file_exists = os.path.isfile(log_file)
    #     with open(log_file, mode="a", newline="") as f:
    #         writer = csv.writer(f)
    #         if not file_exists:
    #             header = ["step", "avg_lpips", "avg_pe", "avg_fid"]
    #             for scale in scales:
    #                 header += [f"{scale}_lpips", f"{scale}_pe", f"{scale}_fid"]
    #             writer.writerow(header)
    #         row = [global_step, avg_lpips, avg_pe, avg_fid]
    #         for scale in scales:
    #             r = val_results[scale]
    #             row += [r["lpips"], r["pe"], r["fid"]]
    #         writer.writerow(row)

    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     return best_lpips, logs


    def evaluate_val_set(model, dl_list, net_lpips, pe_loss_fn, args, global_step,
                         best_lpips, logs, save_vis=False):
        """
        多卡推理 + 主卡 FID 评估。每张卡处理自己分到的样本，
        主卡读完整目录算 FID，LPIPS/PE 通过 gather_for_metrics 聚合。
        """
        gc.collect()
        torch.cuda.empty_cache()
        model.eval()
        device = accelerator.device
        rank = accelerator.process_index

        _all_scales = ['40x', '20x', '40x_X8', '20x_X8']
        scales = _all_scales[:len(dl_list)]
        val_results = {s: {} for s in scales}
        all_lpips, all_pe, all_fid = [], [], []

        for scale, dl in zip(scales, dl_list):
            # 临时目录（共享盘 + rank 隔离的文件名）
            gt_dir = Path(args.output_dir, "temp_eval", f"step{global_step}", scale, "gt")
            sr_dir = Path(args.output_dir, "temp_eval", f"step{global_step}", scale, "sr")
            if accelerator.is_main_process:
                gt_dir.mkdir(parents=True, exist_ok=True)
                sr_dir.mkdir(parents=True, exist_ok=True)
            accelerator.wait_for_everyone()

            l_lpips, l_pe = [], []
            vis_count = 0

            for step, batch in enumerate(dl):
                x_src = batch["conditioning_pixel_values"].to(device)
                x_tgt = batch["output_pixel_values"].to(device)
                B = x_src.shape[0]

                with torch.no_grad():
                    batch["prompt"] = [batch["pos_prompt"][0] + ", " + args.pos_prompt for _ in range(B)]
                    x_tgt_pred, _, _ = model(x_src, batch=batch, prompt=batch["prompt"], args=args)

                    # ----- LPIPS -----
                    loss_lpips = net_lpips(x_tgt_pred, x_tgt).mean()
                    l_lpips.append(accelerator.gather_for_metrics(loss_lpips.detach().unsqueeze(0)))

                    # ----- PE -----
                    if scale == '20x' or scale == '20x_X8':
                        loss_pe = pe_loss_fn(x_tgt_pred, x_tgt, mag_prompt="20× magnification")
                    else:
                        loss_pe = pe_loss_fn(x_tgt_pred, x_tgt)
                    l_pe.append(accelerator.gather_for_metrics(loss_pe.detach().unsqueeze(0)))

                    # ----- 落盘用于 FID（每张卡只存自己的） -----
                    for i in range(B):
                        fname = f"r{rank}_s{step}_b{i}.png"
                        transforms.ToPILImage()(x_tgt[i].cpu().clamp(0, 1)).save(gt_dir / fname)
                        transforms.ToPILImage()(x_tgt_pred[i].cpu().clamp(0, 1)).save(sr_dir / fname)

                    # ----- 可视化（仅主卡） -----
                    if save_vis and accelerator.is_main_process and vis_count < 5:
                        to_vis = lambda x: x.cpu().detach() * 0.5 + 0.5
                        combined = torch.cat([to_vis(x_src), to_vis(x_tgt_pred),
                                              to_vis(torch.clamp(x_tgt, -1, 1))], dim=3)
                        outf = os.path.join(args.output_dir, f"eval/val_{scale}_{step}.png")
                        transforms.ToPILImage()(combined[0]).save(outf)
                        vis_count += 1

            # ============ 关键同步点：等所有卡落盘完成 ============
            accelerator.wait_for_everyone()

            # ============ FID 只在主卡上对完整目录计算 ============
            if accelerator.is_main_process:
                fid_value = calculate_fid_given_paths(
                    [str(gt_dir), str(sr_dir)],
                    batch_size=8, device=str(device), dims=2048
                )
                # LPIPS / PE 聚合（gather_for_metrics 已经处理过去重）
                lpips_mean = torch.cat(l_lpips).mean().item()
                pe_mean = torch.cat(l_pe).mean().item()

                val_results[scale] = {"lpips": lpips_mean, "pe": pe_mean, "fid": fid_value}
                all_lpips.append(lpips_mean)
                all_pe.append(pe_mean)
                all_fid.append(fid_value)

                # 清理本 scale 临时目录
                shutil.rmtree(gt_dir.parent, ignore_errors=True)

            accelerator.wait_for_everyone()

        # ============ 汇总日志 + 保存最佳（仅主卡） ============
        if accelerator.is_main_process:
            avg_lpips = float(np.mean(all_lpips))
            avg_pe = float(np.mean(all_pe))
            avg_fid = float(np.mean(all_fid))

            logs["val/avg_lpips"] = avg_lpips
            logs["val/avg_pe"] = avg_pe
            logs["val/avg_fid"] = avg_fid
            for scale in scales:
                r = val_results[scale]
                logs[f"val/{scale}_lpips"] = r["lpips"]
                logs[f"val/{scale}_pe"] = r["pe"]
                logs[f"val/{scale}_fid"] = r["fid"]

            print(f"[Step {global_step}] Avg: LPIPS={avg_lpips:.4f}, PE={avg_pe:.4f}, FID={avg_fid:.4f}")
            for scale in scales:
                r = val_results[scale]
                print(f"  [{scale.upper()}] LPIPS={r['lpips']:.4f}, PE={r['pe']:.4f}, FID={r['fid']:.4f}")

            # 保存最佳模型
            if avg_lpips < best_lpips:
                best_lpips = avg_lpips
                outm = os.path.join(args.output_dir, "checkpoints",
                                    f"bestmodel_{global_step}_{round(avg_lpips, 4)}_{round(avg_pe, 4)}.pkl")
                accelerator.unwrap_model(model).save_model(outm)

            # 写 CSV
            log_file = os.path.join(args.output_dir, "eval/val_log.csv")
            file_exists = os.path.isfile(log_file)
            with open(log_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    header = ["step", "avg_lpips", "avg_pe", "avg_fid"]
                    for s in scales:
                        header += [f"{s}_lpips", f"{s}_pe", f"{s}_fid"]
                    writer.writerow(header)
                row = [global_step, avg_lpips, avg_pe, avg_fid]
                for s in scales:
                    r = val_results[s]
                    row += [r["lpips"], r["pe"], r["fid"]]
                writer.writerow(row)

            # 清理根临时目录
            shutil.rmtree(Path(args.output_dir, "temp_eval", f"step{global_step}"), ignore_errors=True)

        # 广播 best_lpips 保持所有 rank 状态一致
        best_lpips_t = torch.tensor([best_lpips], device=device)
        if accelerator.num_processes > 1:
            torch.distributed.broadcast(best_lpips_t, src=0)
        best_lpips = best_lpips_t.item()

        accelerator.wait_for_everyone()
        gc.collect()
        torch.cuda.empty_cache()
        model.train()
        return best_lpips, logs

    # start the training loop
    global_step = 0
    best_lpips = 1e10

    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            m_acc = [model_gen, net_disc]
            with accelerator.accumulate(*m_acc):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                B, C, H, W = x_src.shape
                # get text prompts from GT
                batch["pos_prompt"] = [batch["pos_prompt"][i] + ", " + args.pos_prompt for i in range(B)]
                batch["neg_prompt"] = [batch["neg_prompt"][i] + ", " + args.neg_prompt for i in range(B)]
                mag_prompt = batch["mag_prompt"]
                neg_probs = torch.rand(B).to(accelerator.device)

                # build mixed prompt and target
                mixed_tag_prompt = [_neg_tag if p_i < args.neg_prob else _pos_tag for _neg_tag, _pos_tag, p_i in
                                    zip(batch["neg_prompt"], batch["pos_prompt"], neg_probs)]
                neg_probs = neg_probs.reshape(B, 1, 1, 1)
                is_mix = neg_probs < args.neg_prob
                mixed_tgt = torch.where(is_mix, x_src, x_tgt)

                # forward pass
                x_tgt_pred, latents_pred, prompt_embeds = model_gen(x_src.detach(), batch=batch,
                                                                    prompt=mixed_tag_prompt, args=args)
                # Reconstruction loss
                loss_l2 = F.mse_loss(x_tgt_pred.float(), mixed_tgt.detach().float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), mixed_tgt.detach().float()).mean() * args.lambda_lpips
                loss_pe = pe_loss_fn(x_tgt_pred.float(), mixed_tgt.detach().float(),
                                     mag_prompt=mag_prompt) * args.lambda_pe
                loss = loss_l2 + loss_lpips + loss_pe

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                if args.gan:
                    """
                    Generator loss: fool the discriminator
                    """
                    x_tgt_pred, latents_pred, prompt_embeds = model_gen(x_src.detach(), batch=batch,
                                                                        prompt=batch["pos_prompt"], args=args)
                    lossG = net_disc(x_tgt_pred, for_G=True).mean() * args.lambda_gan
                    accelerator.backward(lossG)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                    """
                    Discriminator loss: fake image vs real image
                    """
                    lossD_real = net_disc(x_tgt.detach(), for_real=True).mean() * args.lambda_gan
                    accelerator.backward(lossD_real.mean())
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                    optimizer_disc.step()
                    lr_scheduler_disc.step()
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                    lossD_fake = net_disc(x_tgt_pred.detach(), for_real=False).mean() * args.lambda_gan
                    accelerator.backward(lossD_fake.mean())
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                    optimizer_disc.step()
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                    lossD = lossD_real + lossD_fake

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # 所有 rank 都初始化 logs（非主卡传空 dict 给 evaluate_val_set）
                logs = {}

                # 训练 loss 日志和 checkpoint 保存：仅主卡
                if accelerator.is_main_process:
                    if args.gan:
                        logs["loss_D"] = lossD.detach().item()
                        logs["loss_G"] = lossG.detach().item()
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    logs["loss_pe"] = loss_pe.detach().item()
                    progress_bar.set_postfix(**logs)

                    if global_step % args.checkpointing_steps == 1 and global_step >= args.checkpointing_steps:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(model_gen).save_model(outf)

                # eval：所有 rank 都进入，函数内部自己分工
                if global_step % args.checkpointing_steps == 1 and global_step >= args.checkpointing_steps:
                    best_lpips, logs = evaluate_val_set(
                        accelerator.unwrap_model(model_gen),
                        dl_list,
                        net_lpips,
                        pe_loss_fn,
                        args,
                        global_step,
                        best_lpips,
                        logs,
                        save_vis=True,
                    )

                # tensorboard 日志：仅主卡
                if accelerator.is_main_process:
                    accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args()
    main(args)
