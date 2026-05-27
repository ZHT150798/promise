# wsi_level_slide_infer.py
# Supports WSI slide formats via OpenSlide (.svs, .tiff, etc.)
import os, time
import json
import sys
from tqdm import tqdm

sys.path.append(os.getcwd())
import glob, gc, math
import argparse

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
from new_osediff import OSEDiff_test
from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
import random
from patchify import patchify, unpatchify
import pyvips
from my_utils.wsi.wsi_utils import segment_foreground_background
from my_utils.wsi.WholeSlideImage import WholeSlideImage

tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
])


def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, default='preset/datasets/test_dataset/input',
                        help='path to the input image')
    parser.add_argument('--output_dir', '-o', type=str, default='preset/datasets/test_dataset/output',
                        help='the directory to save the output')
    parser.add_argument('--coord_dir', '-c', type=str, default='preset/datasets/test_dataset/seg',
                        help='the directory to save the coord')
    parser.add_argument('--ref_path', '-r', type=str, default='preset/datasets/test_dataset/gt')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=None, help='sd model path')
    parser.add_argument('--mag', type=str, default="40x", help="image magnification", choices=["40x", "20x"])
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--mpp", type=float, default=0.25)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='wavelet')
    parser.add_argument("--osediff_path", type=str, default=None)
    parser.add_argument("--pos_prompt", type=str,
                        default="40× magnification, 4× super-resolution, A high-resolution, Clear tissue borders, Clear morphological features, high magnification.") # 4× super-resolution,
    parser.add_argument("--neg_prompt", type=str,
                        default="blur, low quality,  low resolution, oversmooth, low magnification.")
    parser.add_argument('--save_prompts', type=bool, default=True)
    # precision setting
    parser.add_argument("--mixed_precision", type=str, choices=['fp16', 'fp32'], default="fp16")
    parser.add_argument('--save_format', type=str, default='svs', choices=['png', 'jpg', 'tif', 'svs'],
                        help='Output image format: png/jpg/tif(pyramidal)')
    # merge lora
    parser.add_argument("--merge_and_unload_lora", default=False)  # merge lora weights before inference
    # tile setting
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument('--use_cfg', type=bool, default=True)  ##
    parser.add_argument('--eval', type=bool, default=False)

    args = parser.parse_args()
    set_seeds(3407)
    print(f"Useing CFG:{args.use_cfg}")
    # initialize the model
    model = OSEDiff_test(args)

    # get all input images
    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(f'{args.input_image}/*'))
    else:
        image_names = [args.input_image]

    # weight type
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    # make the output dir
    output_dir = args.output_dir  # os.path.join(args.output_dir, args.align_method)
    os.makedirs(output_dir, exist_ok=True)
    print(f'There are {len(image_names)} slide.')

    # 创建时间记录文件
    log_path = os.path.join(output_dir, 'processing_times.csv')
    if not os.path.isfile(log_path):
        with open(log_path, 'w') as f:
            f.write('image_name,SR_time(s),patchify_time(s),unpatchify_time(s),save_time(s),total_time(s)\n')

    # patch stride
    patch_size = args.patch_size
    stride = patch_size  # 保持原逻辑
    total_sr_time, total_slide = 0.0, 0.0
    total_pre_time, total_post_time = 0.0, 0.0
    # 遍历所有输入图像
    for n in tqdm(range(len(image_names)), desc="Processing Images", position=0, colour='green', leave=True):
        image_name = image_names[n]
        input_ext = os.path.splitext(image_name)[-1].lower()
        ext = args.save_format.lower()

        # 输出路径
        if ext in ['png', 'jpg', 'tif', 'svs']:
            output_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(image_name))[0] + "." + ext)
        else:  # svs/tiff
            output_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(image_name))[0] + ".png")

        # 如果文件已存在，则跳过
        if os.path.isfile(output_path):
            print(f"图像 '{os.path.basename(image_name)}' 已存在，跳过操作。")
            continue

        tic = time.time()  # 开始计时

        # ===== 普通图片处理 =====
        if input_ext in ['.png', '.jpg', '.jpeg']:
            input_image = Image.open(image_name).convert('RGB')
            lr_width, lr_height = input_image.size
            rscale = args.upscale
            mpp = None

            # 转为 numpy 并归一化
            input_image = np.array(input_image).astype(np.float32) / 255.0

            # 上采样到 HR 尺寸
            input_image = cv2.resize(input_image, (lr_width*rscale, lr_height*rscale), interpolation=cv2.INTER_CUBIC)

            # 补齐
            pad_w = stride - (lr_width*rscale - patch_size) % stride
            pad_h = stride - (lr_height*rscale - patch_size) % stride
            input_image = np.pad(input_image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=255)

            # 切片
            image_patches = patchify(input_image, (patch_size, patch_size, 3), step=stride)

        # ===== WSI 处理 =====
        else:
            slide = WholeSlideImage(image_name)
            input_image = slide.get_lr_image()
            hr_w, hr_h = slide.get_hr_size()
            mpp = slide.get_mpp()

            # 转为 numpy 并归一化
            input_image = np.array(input_image).astype(np.float32) / 255.0

            # 上采样到 HR 尺寸
            input_image = cv2.resize(input_image, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)

            # 补齐
            pad_w = stride - (hr_w - patch_size) % stride
            pad_h = stride - (hr_h - patch_size) % stride
            input_image = np.pad(input_image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=255)

            # 切片
            image_patches = patchify(input_image, (patch_size, patch_size, 3), step=stride)
        del input_image

        toc0 = time.time()  # patchify 计时

        # ===== Patch 超分辨率处理 =====
        d1, d2, d3, h, w, c = image_patches.shape
        indices = [(i, j) for i in range(d1) for j in range(d2)]
        pos_tag_prompt = [args.pos_prompt]
        neg_tag_prompt = [args.neg_prompt]
        count = 0

        for i, j in tqdm(indices, desc=f"Patch SR id {n + 1}", position=1, leave=False):
            patch = image_patches[i, j]
            if segment_foreground_background((patch[0, :, :, :] * 255.0).astype(np.uint8)) > args.threshold:
                #去除分布不齐带来的伪影
                final_bg_patch = patch[0]
                final_bg_patch = np.clip(final_bg_patch, 0.0, 1.0).astype(np.float32)
                image_patches[i, j] = final_bg_patch
                continue
            x_tgt = torch.from_numpy(patch).permute(0, 3, 1, 2).to(model.device)
            count += 1
            with torch.no_grad():
                lq = x_tgt * 2 - 1
                output_image = model(lq, pos_prompt=pos_tag_prompt, neg_prompt=neg_tag_prompt, use_cfg=args.use_cfg)[0]
                output_pil = transforms.ToPILImage()(output_image.cpu() * 0.5 + 0.5)
                im_lr_resize = transforms.ToPILImage()(lq[0].cpu() * 0.5 + 0.5)

                # 颜色对齐
                if args.align_method == 'adain':
                    output_pil = adain_color_fix(target=output_pil, source=im_lr_resize)
                elif args.align_method == 'wavelet':
                    output_pil = wavelet_color_fix(target=output_pil, source=im_lr_resize)

            # 转回 numpy 并限制值域
            out_img = torch.from_numpy(np.array(output_pil).astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
            image_patches[i,j] = torch.clamp(out_img, min=0.0, max=1.0).squeeze(0).permute(1,2,0).cpu().numpy()

        print(f"SR patches:{count}/{d1*d2}")
        toc1 = time.time()

        del patch, x_tgt, lq, output_image, output_pil, im_lr_resize, out_img

        # ===== 重建完整图像 =====
        if input_ext in ['.png', '.jpg', '.jpeg']:
            sr = unpatchify(image_patches, (lr_height*rscale + pad_h, lr_width*rscale + pad_w, 3))[:lr_height*rscale, :lr_width*rscale, :]
        else:
            sr = unpatchify(image_patches, (hr_h + pad_h, hr_w + pad_w, 3))[:hr_h, :hr_w, :]

        del image_patches
        sr_image = (sr*255.0).astype(np.uint8)
        toc2 = time.time()
        if ext != 'tif' and ext != 'svs':
            cv2.imwrite(output_path, cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR))
            del sr_image
        else:
            # sr_image 是 numpy，需要转换为 pyvips.Image
            # pyvips 支持从 numpy 转 image: 使用 Image.new_from_memory
            height, width, channels = sr_image.shape
            sr_img_vips = pyvips.Image.new_from_memory(sr_image.tobytes(), width, height, channels, format=pyvips.BandFormat.UCHAR)
            del sr_image

            if mpp is None:
                print(f"Warning: Failed to read MPP for {image_name}, using default value {args.mpp} µm/pixel")
                mpp = args.mpp
                # 估算文件大小（H*W*C bytes）
            estimated_size_bytes = height * width * channels
            bigtiff_needed = estimated_size_bytes > (4 * 1024**3)  # 4GB
            sr_img_vips.tiffsave(
                output_path,
                tile=True,
                pyramid=True,
                compression="jpeg",
                tile_width=256,
                tile_height=256,
                bigtiff=bigtiff_needed,
                xres=1000/mpp,
                yres=1000/mpp,
                # subifd=True,
                properties=True,
                Q=100
            )
            del sr_img_vips


        toc = time.time()
        # 打印时间信息
        total_pre_time += (toc0 - tic)
        total_sr_time += (toc1 - toc0)
        total_post_time += (toc - toc1)
        total_slide += 1
        print(f"SR time: {round(toc1 - toc0, 1)} seconds\n"
              f"patchify time: {round(toc0 - tic, 1)} seconds\n"
              f"unpatchify patches time: {round(toc2 - toc1, 1)} seconds\n"
              f"save slide time: {round(toc - toc2, 1)} seconds\n"
              f"Total time: {round(toc - tic, 1)} seconds\n")
        # 写入时间记录到CSV
        with open(log_path, 'a') as f:
            f.write(f'{os.path.basename(image_name)},'
                    f'{round(toc1 - toc0, 1)},'
                    f'{round(toc0 - tic, 1)},'
                    f'{round(toc2 - toc1, 1)},'
                    f'{round(toc - toc2, 1)},'
                    f'{round(toc - tic, 1)}\n')
        clear_memory()
    with open(log_path, 'a') as f:
        f.write(f'\nAverage preprocess time:{round(total_pre_time / total_slide, 2)}')
        f.write(f'\nAverage SR time:{round(total_sr_time / total_slide, 2)}')
        f.write(f'\nAverage postprocess time:{round(total_post_time / total_slide, 2)}')
    clear_memory()


