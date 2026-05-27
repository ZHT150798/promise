import os
import numpy as np
import cv2
import glob
import math
import yaml
import random
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torchvision import transforms

from basicsr.data.transforms import augment
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.utils import DiffJPEG, USMSharp, img2tensor, tensor2img
from basicsr.utils.img_process_util import filter2D
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize, rgb_to_grayscale)

cur_path = os.path.dirname(os.path.abspath(__file__))


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def opt_parse(opt_path):
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)  # ignore_security_alert_wait_for_fix RCE

    return opt


class RealESRGAN_degradation(object):
    def __init__(self, opt_name='params_realesrgan.yml', device='cpu'):
        opt_path = f'{cur_path}/{opt_name}'
        self.opt = opt_parse(opt_path)
        self.device = device  #torch.device('cpu')
        optk = self.opt['kernel_info']

        # blur settings for the first degradation
        self.blur_kernel_size = optk['blur_kernel_size']
        self.kernel_list = optk['kernel_list']
        self.kernel_prob = optk['kernel_prob']
        self.blur_sigma = optk['blur_sigma']
        self.betag_range = optk['betag_range']
        self.betap_range = optk['betap_range']
        self.sinc_prob = optk['sinc_prob']

        # blur settings for the second degradation
        self.blur_kernel_size2 = optk['blur_kernel_size2']
        self.kernel_list2 = optk['kernel_list2']
        self.kernel_prob2 = optk['kernel_prob2']
        self.blur_sigma2 = optk['blur_sigma2']
        self.betag_range2 = optk['betag_range2']
        self.betap_range2 = optk['betap_range2']
        self.sinc_prob2 = optk['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = optk['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

        self.jpeger = DiffJPEG(differentiable=False).to(self.device)
        self.usm_shaper = USMSharp().to(self.device)

    def color_jitter_pt(self, img, brightness, contrast, saturation, hue):
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img

    def random_augment(self, img_gt):
        # random horizontal flip
        img_gt, status = augment(img_gt, hflip=True, rotation=False, return_status=True)
        """
        # random color jitter 
        if np.random.uniform() < self.opt['color_jitter_prob']:
            jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
            img_gt = img_gt + jitter_val
            img_gt = np.clip(img_gt, 0, 1)    

        # random grayscale
        if np.random.uniform() < self.opt['gray_prob']:
            #img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_RGB2GRAY)
            img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])
        """
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=False, float32=True)[0].unsqueeze(0)

        return img_gt

    def random_augment_pair(self, img_gt, img_lr):
        # random horizontal flip
        p = random.random()
        if p < 0.5:
            cv2.flip(img_gt, 1, img_gt)
            cv2.flip(img_lr, 1, img_lr)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=False, float32=True)[0].unsqueeze(0)
        img_lr = img2tensor([img_lr], bgr2rgb=False, float32=True)[0].unsqueeze(0)
        _, _, h, w = img_gt.size()
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        img_lr = F.interpolate(img_lr, size=(h, w), mode=mode)
        # clamp and round
        img_lr = torch.clamp((img_lr * 255.0).round(), 0, 255) / 255.

        return img_gt, img_lr

    def random_augment_lr_compress(self, img_gt, img_lr):
        _, _, sinc_kernel = self.random_kernels()
        p = random.random()
        # random horizontal flip
        if p < 0.5:
            cv2.flip(img_gt, 1, img_gt)
            cv2.flip(img_lr, 1, img_lr)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=False, float32=True)[0].unsqueeze(0)
        img_lr = img2tensor([img_lr], bgr2rgb=False, float32=True)[0].unsqueeze(0)
        _, _, h, w = img_gt.size()

        # Randomly select a downsampling factor
        scale_factor = random.choice([4, 8])  # 1,
        if scale_factor == 4:
            prompt = "4× super-resolution"
        elif scale_factor == 8:
            prompt = "8× super-resolution"
        # else:
        #     prompt = "16× super-resolution"
        if scale_factor > 4:
            with torch.no_grad():
                # resize
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(img_gt, size=(h // scale_factor, w // scale_factor), mode=mode)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                img_lr = self.jpeger(out, quality=jpeg_p)
        # === Upsample to original resolution ===
        mode_up = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(img_lr, size=(h, w), mode=mode_up)
        # clamp and round
        img_lr = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        # mode = random.choice(['area', 'bilinear', 'bicubic'])
        # img_lr = F.interpolate(out, size=(h, w), mode=mode)

        return img_gt, img_lr, prompt

    def random_augment_wsi_compress(self, img_gt, img_lr):
        """
        对输入 HR/LR 图像对进行随机 mag、scale 和数据增强处理。
        输入：
            img_hr: 1024×1024 的 HR 图像（numpy array）
            img_lr: 256×256 的 LR 图像（numpy array）
            transform: 用于裁剪/resize的函数
        输出：
            img_hr_tensor: 512×512 HR Tensor
            img_lr_tensor: 128×128 LR Tensor
            prompt:  magnification + scale 的提示字符串
        """
        # 随机水平翻转
        p = random.random()
        if p < 0.5:
            cv2.flip(img_gt, 1, img_gt)
            cv2.flip(img_lr, 1, img_lr)

        # === 随机选择 magnification ===
        mag = random.choice(['20x', '40x'])
        if mag == '40x':
            # 中心裁剪：HR 512 / LR 128
            transform_hr = transforms.CenterCrop(512)
            transform_lr = transforms.CenterCrop(128)
            mag_prompt = "40× magnification"
        else:  # 20x
            # resize：HR 512 / LR 128
            transform_hr = transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC)
            transform_lr = transforms.Resize(128, interpolation=transforms.InterpolationMode.BICUBIC)
            mag_prompt = "20× magnification"
            # BGR to RGB, HWC to CHW, numpy to tensor
        # numpy -> tensor
        img_gt = img2tensor([img_gt], bgr2rgb=False, float32=True)[0].unsqueeze(0)
        img_lr = img2tensor([img_lr], bgr2rgb=False, float32=True)[0].unsqueeze(0)
        # 应用 transform
        img_gt = transform_hr(img_gt)
        img_lr = transform_lr(img_lr)
        _, _, h, w = img_gt.size()

        # Randomly select a downsampling factor
        scale_factor = random.choice([4, 8])  # scale_factor = 4 ,
        if scale_factor == 4:
            scale_prompt = "4× super-resolution"
        else:
            scale_prompt = "8× super-resolution"

        # if scale_factor >= 4:
        #     with torch.no_grad():
        #         # resize
        #         mode = 'bicubic' # random.choice(['area', 'bilinear', 'bicubic'])
        #         out = F.interpolate(img_gt, size=(h // scale_factor, w // scale_factor), mode=mode)
        #         # JPEG compression
        #         jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
        #         out = torch.clamp(out, 0, 1)
        #         img_lr = self.jpeger(out, quality=jpeg_p)
        if scale_factor >= 4:
            jpeg_min, jpeg_max = self.opt['jpeg_range2']
            # 候选项：None 表示直接保留原始 LR
            jpeg_candidates = [None] + list(range(jpeg_min, jpeg_max + 1, 5))
            jpeg_quality = random.choice(jpeg_candidates)
            if jpeg_quality is None:
                # 以同样概率直接使用原始 LR
                # 此时 img_lr 保持 transform_lr(img_lr) 后的结果
                pass
            else:
                with torch.no_grad():
                    # resize
                    mode = 'bicubic'
                    out = F.interpolate(
                        img_gt,
                        size=(h // scale_factor, w // scale_factor),
                        mode=mode
                    )
                    # JPEG compression
                    jpeg_p = out.new_full(

                        (out.size(0),),

                        float(jpeg_quality)

                    )
                    out = torch.clamp(out, 0, 1)
                    img_lr = self.jpeger(out, quality=jpeg_p)
        # # === Upsample to original resolution ===
        mode_up = 'bicubic' # random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(img_lr, size=(h, w), mode=mode_up)
        # clamp and round
        img_lr = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        # === 合并 prompt ===
        prompt = f"{mag_prompt}, {scale_prompt}"

        return img_gt, img_lr, prompt, mag_prompt

    def random_augment_pair_infer(self, img_gt, img_lr):
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=False, float32=True)[0].unsqueeze(0)
        img_lr = img2tensor([img_lr], bgr2rgb=False, float32=True)[0].unsqueeze(0)
        _, _, h, w = img_gt.size()
        _, _, h1, w1 = img_lr.size()
        img_lr = F.interpolate(img_lr, size=(h, w))
        # clamp and round
        img_lr = torch.clamp((img_lr * 255.0).round(), 0, 255) / 255.
        scale_factor = h // h1
        if scale_factor == 4:
            prompt = "4× super-resolution"
        elif scale_factor == 8:
            prompt = "8× super-resolution"
        else:
            prompt = "16× super-resolution"

        return img_gt, img_lr, prompt

    def random_augment_multi(self, img_gt, img_lr, mag='40x', scale_factor=4):

        # numpy -> tensor
        img_gt = img2tensor([img_gt], bgr2rgb=False, float32=True)[0].unsqueeze(0)
        img_lr = img2tensor([img_lr], bgr2rgb=False, float32=True)[0].unsqueeze(0)
        _, _, h, w = img_gt.size()
        _, _, h1, w1 = img_lr.size()
        if mag == '40x':
            transform_hr = transforms.CenterCrop(h // 2)
            transform_lr = transforms.CenterCrop(h1 // 2)
            mag_prompt = "40× magnification"
        else:  # 20x
            transform_hr = transforms.Resize(h // 2, interpolation=transforms.InterpolationMode.BICUBIC)
            transform_lr = transforms.Resize(h1 // 2, interpolation=transforms.InterpolationMode.BICUBIC)
            mag_prompt = "20× magnification"
        # 应用 transform
        img_gt = transform_hr(img_gt)
        img_lr = transform_lr(img_lr)

        img_lr = F.interpolate(img_lr, size=(h // 2, w // 2))
        # clamp and round
        img_lr = torch.clamp((img_lr * 255.0).round(), 0, 255) / 255.

        scale_factor = h // h1
        if scale_factor == 4:
            scale_prompt = "4× super-resolution"
        elif scale_factor == 8:
            scale_prompt = "8× super-resolution"
        else:
            raise ValueError(f"Unsupported scale_factor: {scale_factor}")

        # === 合并 prompt ===
        prompt = f"{mag_prompt}, {scale_prompt}"

        return img_gt, img_lr, prompt

    def random_augment_norm(self, img_gt):
        # random horizontal flip
        p = random.random()
        if p < 0.5:
            cv2.flip(img_gt, 1, img_gt)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=False, float32=True)[0].unsqueeze(0)
        _, _, h, w = img_gt.size()
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        img_lr = F.interpolate(img_gt, size=(h // 4, w // 4), mode=mode)
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        img_lr = F.interpolate(img_lr, size=(h, w), mode=mode)
        # clamp and round
        img_lr = torch.clamp((img_lr * 255.0).round(), 0, 255) / 255.

        return img_gt, img_lr

    def random_kernels(self):
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- sinc kernel ------------------------------------- #
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return kernel, kernel2, sinc_kernel

    @torch.no_grad()
    def degrade_process(self, img_gt, resize_bak=False):
        img_gt = self.random_augment(img_gt)
        kernel1, kernel2, sinc_kernel = self.random_kernels()
        img_gt, kernel1, kernel2, sinc_kernel = img_gt.to(self.device), kernel1.to(self.device), kernel2.to(
            self.device), sinc_kernel.to(self.device)
        #img_gt = self.usm_shaper(img_gt) # shaper gt
        ori_h, ori_w = img_gt.size()[2:4]

        #scale_final = random.randint(4, 16)
        scale_final = 4

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(img_gt, kernel1)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.opt['resize_range'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.opt['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # noise
        gray_noise_prob = self.opt['gray_noise_prob']
        if np.random.uniform() < self.opt['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.opt['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
        out = torch.clamp(out, 0, 1)
        out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.opt['second_blur_prob']:
            out = filter2D(out, kernel2)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.opt['resize_range2'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.opt['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out, size=(int(ori_h / scale_final * scale), int(ori_w / scale_final * scale)), mode=mode)
        # noise
        gray_noise_prob = self.opt['gray_noise_prob2']
        if np.random.uniform() < self.opt['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.opt['poisson_scale_range2'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // scale_final, ori_w // scale_final), mode=mode)
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // scale_final, ori_w // scale_final), mode=mode)
            out = filter2D(out, sinc_kernel)

        if np.random.uniform() < self.opt['gray_prob']:
            out = rgb_to_grayscale(out, num_output_channels=1)

        if np.random.uniform() < self.opt['color_jitter_prob']:
            brightness = self.opt.get('brightness', (0.5, 1.5))
            contrast = self.opt.get('contrast', (0.5, 1.5))
            saturation = self.opt.get('saturation', (0, 1.5))
            hue = self.opt.get('hue', (-0.1, 0.1))
            out = self.color_jitter_pt(out, brightness, contrast, saturation, hue)

        if resize_bak:
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h, ori_w), mode=mode)
        # clamp and round
        img_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        return img_gt, img_lq
