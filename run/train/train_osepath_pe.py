import os
import lpips
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from tqdm.auto import tqdm
from torchvision import transforms
import diffusers
import gc
import timm
import numpy as np
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from new_osediff import OSEDiff_gen
from dataloaders.realsr_dataset import WSISRDataset, WSISRDataset_val

from pathlib import Path
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs
from my_utils.fd_loss.conch import build_conch
from my_utils.fd_loss import pe_loss


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

    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)

    # training details
    parser.add_argument("--gan_disc_type", default="vagan")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--output_dir", default='experience/osepath')
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--num_samples_eval", type=int, default=500, help="Number of samples to use for all evaluation")
    parser.add_argument("--max_train_steps", type=int, default=100000,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=8,)
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
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)
    parser.add_argument("--logging_dir", type=str, default="logs")
    
    
    parser.add_argument("--tracker_project_name", type=str, default="train_osediff", help="The name of the wandb project to log to.")
    parser.add_argument('--train_path', type=str, default='YOUR TXT FILE PATH', help='A comma-separated list of integers')
    parser.add_argument('--val_path', type=str, default='/media/dell/data/zhangv1/WSISR/WSI_Level_SR/wsisr_pipeline/preset/datasets/tcga_l/val/patch/',
                        help='A comma-separated list of integers')
    parser.add_argument("--deg_type", default="lr", type=str, choices=["norm", "realgan", "lr"])
    parser.add_argument("--deg_file_path", default="params_realesrgan.yml", type=str)
    parser.add_argument("--pretrained_model_name_or_path", default=None, type=str)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_lpips", default=1.0, type=float)
    parser.add_argument("--lambda_pe", default=1.0, type=float)
    parser.add_argument("--pos_prompt", type=str, default="A high-resolution, Clear tissue borders, Clear morphological features, high magnification.")
    parser.add_argument("--neg_prompt", type=str, default="blur, low quality,  low resolution, oversmooth, low magnification.")

    # lora setting
    parser.add_argument("--lora_rank_unet", default=4, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)
    parser.add_argument("--neg_prob", default=0.05, type=float)

    parser.add_argument("--gan", action="store_true", help="use_gan")
    parser.add_argument("--pe_type", default="conch", type=str, choices=["conch", "gigapath", "uni"])
    parser.add_argument("--osediff_path", type=str, default="/media/dell/data/zhangv1/WSISR/WSI_Level_SR/wsisr_pipeline/experience/wsisr_newpe/checkpoints_nogan/model_4001_bestpe.pkl")



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
        os.makedirs(os.path.join(args.output_dir, "checkpoints_nogan"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    model_gen = OSEDiff_gen(args)
    model_gen.set_train()
    total_params = sum(p.numel() for p in model_gen.parameters())
    trainable_params = sum(p.numel() for p in model_gen.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Pe_type: {args.pe_type}")


    if args.gan_disc_type == "vagan":
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(cv_type='dino', output_type='conv_multi_level', loss_type=args.gan_loss_type, device="cuda")
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
        model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
        model.load_state_dict(torch.load("/media/dell/data/zhangv1/WSISR/huggingface/UNI/pytorch_model.bin", map_location="cpu"), strict=True)
        pe_loss_fn = pe_loss.PELoss_uni(model.cuda())
    elif pe_type == "gigapath":
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False, checkpoint_path="/media/dell/data/zhangv1/WSISR/huggingface/prov-gigapath/pytorch_model.bin")
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
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)
    optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
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
    dataset_val = WSISRDataset_val(split="test", args=args)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # Prepare everything with our `accelerator`.
    # model_gen, model_reg, optimizer, optimizer_reg, dl_train, lr_scheduler, lr_scheduler_reg = accelerator.prepare(
    #     model_gen, model_reg, optimizer, optimizer_reg, dl_train, lr_scheduler, lr_scheduler_reg
    # )
    model_gen, net_disc, optimizer, optimizer_disc, dl_train, dl_val, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
        model_gen, net_disc, optimizer, optimizer_disc, dl_train, dl_val, lr_scheduler, lr_scheduler_disc
    )
    net_lpips = accelerator.prepare(net_lpips)
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
        disable=not accelerator.is_local_main_process,)

    # start the training loop
    global_step = 0
    best_l2 = 1e10

    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            m_acc = [model_gen, net_disc]
            with accelerator.accumulate(*m_acc):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                B, C, H, W = x_src.shape
                # get text prompts from GT
                batch["pos_prompt"] = [args.pos_prompt for _ in range(B)]
                batch["neg_prompt"]= [args.neg_prompt for _ in range(B)]
                neg_probs = torch.rand(B).to(accelerator.device)

                # build mixed prompt and target
                mixed_tag_prompt = [_neg_tag if p_i < args.neg_prob else _pos_tag for _neg_tag, _pos_tag, p_i in zip(batch["neg_prompt"], batch["pos_prompt"], neg_probs)]
                neg_probs = neg_probs.reshape(B, 1, 1, 1)
                is_mix = neg_probs < args.neg_prob
                mixed_tgt = torch.where(is_mix, x_src, x_tgt)

                # forward pass
                x_tgt_pred, latents_pred, prompt_embeds= model_gen(x_src.detach(), batch=batch, prompt=mixed_tag_prompt, args=args)
                # Reconstruction loss
                loss_l2 = F.mse_loss(x_tgt_pred.float(), mixed_tgt.detach().float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), mixed_tgt.detach().float()).mean() * args.lambda_lpips
                loss_pe = pe_loss_fn(x_tgt_pred.float(), mixed_tgt.detach().float()) * args.lambda_pe
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
                    x_tgt_pred, latents_pred, prompt_embeds = model_gen(x_src.detach(), batch=batch, prompt=batch["pos_prompt"], args=args)
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
                    # real image
                    lossD_real = net_disc(x_tgt.detach(), for_real=True).mean() * args.lambda_gan
                    accelerator.backward(lossD_real.mean())
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                    optimizer_disc.step()
                    lr_scheduler_disc.step()
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                    # fake image
                    lossD_fake = net_disc(x_tgt_pred.detach(), for_real=False).mean() * args.lambda_gan
                    accelerator.backward(lossD_fake.mean())
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                    optimizer_disc.step()
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                    lossD = lossD_real + lossD_fake

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    
                    logs = {}
                    # log all the losses
                    if args.gan:
                        logs["loss_D"] = lossD.detach().item()
                        logs["loss_G"] = lossG.detach().item()
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    logs["loss_pe"] = loss_pe.detach().item()
                    progress_bar.set_postfix(**logs)

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1 and global_step>=args.checkpointing_steps:
                        outf = os.path.join(args.output_dir, "checkpoints_nogan", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(model_gen).save_model(outf)
                    # compute validation set FID, L2, LPIPS, CLIP-SIM
                    if global_step % args.checkpointing_steps == 1 and global_step>=args.checkpointing_steps:
                        model_gen.eval()
                        l_l2, l_lpips, l_pe = [], [], []

                        val_count = 0
                        for step, batch_val in enumerate(dl_val):
                            # if step >= 500:
                            #     break
                            x_src = batch_val["conditioning_pixel_values"]
                            x_tgt = batch_val["output_pixel_values"]
                            B, C, H, W = x_src.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                batch_val["prompt"] = [args.pos_prompt for _ in range(B)]
                                x_tgt_pred, _, _ = accelerator.unwrap_model(model_gen)(x_src.detach(), batch=batch_val, prompt=batch["pos_prompt"], args=args)
                                # compute the reconstruction losses
                                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.detach().float(), reduction="mean")
                                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.detach().float()).mean()
                                loss_pe = pe_loss_fn(x_tgt_pred.float(), x_tgt.detach().float())

                                l_l2.append(loss_l2.item())
                                l_lpips.append(loss_lpips.item())
                                l_pe.append(loss_pe.item())

                            if val_count < 5:
                                x_src = x_src.cpu().detach() * 0.5 + 0.5
                                x_tgt = x_tgt.cpu().detach() * 0.5 + 0.5
                                x_tgt_pred = x_tgt_pred.cpu().detach() * 0.5 + 0.5

                                combined = torch.cat([x_src, x_tgt_pred, x_tgt], dim=3)
                                output_pil = transforms.ToPILImage()(combined[0])
                                outf = os.path.join(args.output_dir, f"val_{step}.png")
                                output_pil.save(outf)
                                val_count += 1

                        logs["val/l2"] = np.mean(l_l2)
                        logs["val/lpips"] = np.mean(l_lpips)
                        logs["val/pe"] = np.mean(l_pe)
                        if np.mean(l_lpips) < best_l2:
                            best_l2 = np.mean(l_lpips)
                            outm = os.path.join(args.output_dir, "checkpoints_nogan",
                                                f"bestmodel_{global_step}_{round(np.mean(l_lpips), 4)}_{round(np.mean(l_pe), 4)}.pkl")
                            accelerator.unwrap_model(model_gen).save_model(outm)
                        print(
                            f"Global step:{global_step} [val_lpips={round(np.mean(l_lpips), 4)}, val_l2={round(np.mean(l_l2), 4)}, val_pe={round(np.mean(l_pe), 4)}]")
                        gc.collect()
                        torch.cuda.empty_cache()

                    accelerator.log(logs, step=global_step)

if __name__ == "__main__":
    args = parse_args()
    main(args)


