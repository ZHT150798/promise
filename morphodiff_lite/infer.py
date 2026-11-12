"""Patch-level super-resolution with a distilled MorphoDiff-Lite student.

Adapted from `test.py` of AdcSR (https://github.com/Guaishou74851/AdcSR, Apache-2.0).
The student needs no text encoder and no VAE encoder; the SD-Turbo U-Net is
loaded only to instantiate the architecture, all student weights come from
`--student_ckpt`.

Run from this directory:
    python infer.py -i /path/to/lr_patches -o /path/to/sr_patches --student_ckpt /path/to/net_params_200.pkl
"""
import os, glob, copy, time, random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model import Net
from build import build_half_decoder, load_half_decoder_weights, decoder_tail


def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def build_student(unet, decoder, student_ckpt=None):
    """Assemble PixelUnshuffle + pruned U-Net + pruned decoder (pixels out)."""
    model = Net(unet, copy.deepcopy(decoder))
    if student_ckpt is not None:
        state_dict = torch.load(student_ckpt, map_location="cpu", weights_only=False)
        # checkpoints saved from a DataParallel/DDP-wrapped model carry a "module." prefix
        state_dict = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
    return torch.nn.Sequential(model, *decoder_tail(decoder))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, required=True,
                        help='low-resolution patch (png/jpg) or a directory of patches')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='directory to save the reconstructed patches')
    parser.add_argument('--student_ckpt', type=str, required=True,
                        help='distilled student weights, e.g. weight/net_params_200.pkl')
    parser.add_argument('--sd_turbo_path', type=str, default='stabilityai/sd-turbo',
                        help='SD-Turbo base model (local directory or Hugging Face id)')
    parser.add_argument('--halfdecoder_ckpt', type=str, default='./weight/pretrained/halfDecoder.ckpt')
    parser.add_argument('--upscale', type=int, default=4, choices=[4, 8])
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args(argv)


def main(args):
    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(args.sd_turbo_path, subfolder="unet")
    decoder = load_half_decoder_weights(build_half_decoder(), args.halfdecoder_ckpt)
    model = build_student(unet, decoder, args.student_ckpt).to(device).eval()

    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(os.path.join(args.input_image, '*.jpg')) +
                             glob.glob(os.path.join(args.input_image, '*.png')))
    else:
        image_names = [args.input_image]
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'There are {len(image_names)} images.')

    total_sr_time = 0.0  # forward time only
    with torch.no_grad():
        for image_name in tqdm(image_names, desc="Processing images", unit="image"):
            LR = Image.open(image_name).convert("RGB")
            LR_tensor = transforms.ToTensor()(LR).to(device).unsqueeze(0) * 2 - 1
            if args.upscale == 8:
                # the student itself is a x4 network; x8 inputs are first upsampled x2 (bicubic)
                LR_tensor = F.interpolate(LR_tensor, scale_factor=2, mode="bicubic")

            if device.type == "cuda":
                torch.cuda.synchronize()
            start_sr = time.time()
            SR = model(LR_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_sr_time += time.time() - start_sr

            # match per-channel mean/std of the output to the input
            SR = (SR - SR.mean(dim=[2, 3], keepdim=True)) / SR.std(dim=[2, 3], keepdim=True) \
                 * LR_tensor.std(dim=[2, 3], keepdim=True) + LR_tensor.mean(dim=[2, 3], keepdim=True)
            SR = transforms.ToPILImage()((SR[0] / 2 + 0.5).clamp(0, 1).cpu())
            bname = os.path.basename(image_name)
            SR.save(os.path.join(args.output_dir, os.path.splitext(bname)[0] + ".png"))

    print(f"Average SR forward time per image: {total_sr_time / max(1, len(image_names)):.4f} seconds")


if __name__ == "__main__":
    main(parse_args())
