# -*- coding: utf-8 -*-
"""
WSI HR-LR Image Pair Extraction

This script extracts high-resolution (HR) and corresponding low-resolution (LR)
image pairs from Whole Slide Images (WSI) at a given magnification level.

Author: [Hongtai Zhang]
Date: [2025-03-08]
Version: 1.0

Description:
    - Reads WSI files and extracts patches at specified magnification levels.
    - Generates paired HR-LR images by downsampling HR images to the target LR scale.
    - Saves the extracted image pairs for use in super-resolution tasks.
"""
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time

from my_utils.wsi.WholeSlideImage import WholeSlideImage
from my_utils.wsi.wsi_utils import get_svs_files
from my_utils.fd_loss.conch import build_conch
from scripts.UslClusterSample_pebased import PELoss
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default="/media/dell/data/zhangv1/WSISR/WSI_Level_SR/wsisr_pipeline/preset/datasets/tcga_l/test/wsi",
                    help='path to folder containing raw wsi image files')
parser.add_argument('--seg_path', type=str, default="/media/dell/data/zhangv1/WSISR/WSI_Level_SR/wsisr_pipeline/preset/datasets/tcga_l/test/seg",
                    help='path to folder containing seg result')
parser.add_argument('--save_path', type=str, default="/media/dell/data/zhangv1/WSISR/WSI_Level_SR/wsisr_pipeline/preset/datasets/tcga_l/test/slide",
                    help='path to folder saving patch')


if __name__ == '__main__':
    args = parser.parse_args()
    #conch, eval_transform = build_conch()  # 用户自行定义构建方式
    #pe_loss_fn = PELoss(conch.cuda(), eval_transform)  # 转移到 GPU（如有需要）

    save_dir = args.save_path
    # if os.path.isdir(save_dir):
    #     shutil.rmtree(save_dir)
    os.makedirs(save_dir + "/hr_v2", exist_ok=True)
    #os.makedirs(save_dir + "/lr", exist_ok=True)


    wsi_files = get_svs_files(args.path)
    # 获取hr_lr pairs
    for wsi_file in wsi_files:
        start = time.time()
        wsi = WholeSlideImage(wsi_file)
        wsi_name = wsi.name
        hr, lr = wsi.get_hrlr_pair()
        #hr = hr.resize(0.25)
        # hr.save(f'{save_dir}/hr_v2/{wsi_name}.png')
        #cv2.imwrite(f'{save_dir}/lr/{wsi_name}.png', cv2.cvtColor(lr, cv2.COLOR_RGB2BGR))
        hr.write_to_file(f'{save_dir}/hr/{wsi_name}.png')
        lr.write_to_file(f'{save_dir}/lr/{wsi_name}.png')
        end = time.time()
        print(f"Processing {wsi_name} took {round((end-start), 2)} seconds")