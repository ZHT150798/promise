import os
import sys

sys.path.append(os.getcwd())
import glob
import argparse
import torch
from torchvision import transforms
import csv
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from PIL import Image
import time
from tqdm import tqdm
import numpy as np
from new_osediff import OSEDiff_test
from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
import pytorch_fid.fid_score as fid_score
from my_utils.fd_loss.conch import build_conch
from my_utils.fd_loss import pe_loss
import lpips
import random
from my_utils.metrics import psnr, lpips_score, pe_score

tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, default='/media/dell/T7 Shield/wsisr_pipeline/dataset/multi_mag/internal_tcga/lr_X8',
                        help='path to the input image')
    parser.add_argument('--output_dir', '-o', type=str, default='/media/dell/data/zhangv1/WSISR/method_comparison/patch-multimag/album/only4_8/tcga/8',
                        help='the directory to save the output')
    parser.add_argument('--ref_path', '-r', type=str, default='/media/dell/T7 Shield/wsisr_pipeline/dataset/multi_mag/internal_tcga/hr')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=None, help='sd model path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=8)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='wavelet') #
    parser.add_argument("--osediff_path", type=str, default=None) # 20× magnification, 4× super-resolution, 40× magnification, 4× super-resolution,
    parser.add_argument("--pos_prompt", type=str, default="A high-resolution, Clear tissue borders, Clear morphological features, high magnification.")
    parser.add_argument("--neg_prompt", type=str, default="blur, low quality,  low resolution, oversmooth, low magnification.")
    parser.add_argument('--ram_path', type=str, default=None)
    parser.add_argument('--ram_ft_path', type=str, default=None)
    parser.add_argument('--save_prompts', type=bool, default=True)
    # precision setting
    parser.add_argument("--mixed_precision", type=str, choices=['fp16', 'fp32'], default="fp16")
    # merge lora
    parser.add_argument("--merge_and_unload_lora", default=False)  # merge lora weights before inference
    # tile setting
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    parser.add_argument('--use_cfg', type=bool, default=True)

    args = parser.parse_args()
    set_seeds(3407)
    print(f"Useing CFG:{args.use_cfg}")
    # initialize the model
    model = OSEDiff_test(args)
    conch, eval_transform = build_conch()
    conch.requires_grad_(False)
    pe_loss_fn = pe_loss.PELoss(conch).to(model.device)
    lpips_loss = lpips.LPIPS(net='vgg').to(model.device)

    # get all input images
    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(os.path.join(args.input_image, '*.jpg')) + glob.glob(os.path.join(args.input_image, '*.png')))
    else:
        image_names = [args.input_image]


    # make the output dir
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'There are {len(image_names)} images.')
    tic = time.time()
    total_sr_time = 0.0
    for image_name in tqdm(image_names, desc="Processing images", unit="image"):
        # make sure that the input image is a multiple of 8
        input_image = Image.open(image_name).convert('RGB')
        ori_width, ori_height = input_image.size
        rscale = args.upscale
        resize_flag = False
        if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
            scale = (args.process_size // rscale) / min(ori_width, ori_height)
            input_image = input_image.resize((int(scale * ori_width), int(scale * ori_height)))
            resize_flag = True
        input_image = input_image.resize((input_image.size[0] * rscale, input_image.size[1] * rscale), Image.BICUBIC)

        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        bname = os.path.basename(image_name)

        # get caption
        # validation_prompt, lq = get_validation_prompt(args, input_image, DAPE)
        # if args.save_prompts:
        #     txt_save_path = f"{txt_path}/{bname.split('.')[0]}.txt"
        #     with open(txt_save_path, 'w', encoding='utf-8') as f:
        #         f.write(validation_prompt)
        #         f.close()
        # print(f"process {image_name}, tag: {validation_prompt}".encode('utf-8'))

        # translate the image
        lq = tensor_transforms(input_image).unsqueeze(0).to("cuda")
        with torch.no_grad():
            lq = lq * 2 - 1
            start_sr = time.time()
            output_image = model(lq, pos_prompt=args.pos_prompt, neg_prompt=args.neg_prompt, use_cfg=args.use_cfg)
            torch.cuda.synchronize()
            end_sr = time.time()
            total_sr_time += (end_sr - start_sr)
            output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
            if args.align_method == 'adain':
                output_pil = adain_color_fix(target=output_pil, source=input_image)
            elif args.align_method == 'wavelet':
                output_pil = wavelet_color_fix(target=output_pil, source=input_image)
            else:
                pass
            if resize_flag:
                output_pil.resize((int(args.upscale * ori_width), int(args.upscale * ori_height)))

        #output_pil = output_pil.resize((775, 522), Image.LANCZOS)
        output_pil.save(os.path.join(args.output_dir, os.path.splitext(bname)[0]+".png"))  # os.path.splitext(bname)[0]+".jpg"

    toc = time.time()
    avg_sr_time = total_sr_time / len(image_names)
    print(f"Average SR inference time per image: {avg_sr_time:.3f} seconds")
    print(f"Total elapsed time (including I/O and post-processing): {toc - tic:.3f} seconds")
    indir = args.ref_path
    fid = fid_score.calculate_fid_given_paths([indir, args.output_dir],
                                              batch_size=10, device=model.device,
                                              dims=2048,
                                              num_workers=0)
    psnr_value, std_psnr = psnr(indir, args.output_dir, std=True)
    lpips_value, std_lpips = lpips_score(indir, args.output_dir, lpips_loss, model.device, std=True)
    pe_value, std_pe = pe_score(indir, args.output_dir, pe_loss_fn, model.device, std=True)

    print(f"Average FID score: {fid:.3f}\n"
          f"Average PSNR score: {psnr_value:.3f} ± {std_psnr:.3f}\n"
          f"Average LPIPS score: {lpips_value:.4f} ± {std_lpips:.4f}\n"
          f"Average PEloss score: {pe_value:.4f} ± {std_pe:.4f}\n")
    print(f"Total time: {round(toc - tic, 3)} seconds\n"
          f"Average time per image: {round((toc - tic) / len(image_names), 3)} seconds")

    csv_path = os.path.join(args.output_dir, "metrics.csv")

    # 定义字段名
    fieldnames = [
        "FID", "PSNR", "PSNR_std", "LPIPS", "LPIPS_std", "PE", "PE_std",
        "Total_time(s)", "Avg_time_per_image(s)"
    ]

    # 准备数据记录
    record = {
        "FID": round(fid, 3),
        "PSNR": round(psnr_value, 3),
        "PSNR_std": round(std_psnr, 3),
        "LPIPS": round(lpips_value, 4),
        "LPIPS_std": round(std_lpips, 4),
        "PE": round(pe_value, 4),
        "PE_std": round(std_pe, 4),
        "Total_time(s)": round(toc - tic, 3),
        "Avg_time_per_image(s)": round((toc - tic) / len(image_names), 3)
    }

    # 写入 CSV 文件（如果文件不存在则写入标题）
    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(record)