import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import shutil
import sys, os
from UslClusterSample_pebased import PELoss
sys.path.append("/media/dell/data/zhangv1/WSISR/WSI_Level_SR/wsisr_pipeline")
from my_utils.wsi.WholeSlideImage import WholeSlideImage
from my_utils.wsi.wsi_utils import getPatchFromWsiRoi_PeCluster, getCoords, get_wsi_files
from my_utils.fd_loss.conch import build_conch




parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str,
                    help='path to folder containing raw wsi image files')
parser.add_argument('--seg_path', type=str,
                    help='path to folder containing seg result')
parser.add_argument('--save_path', type=str,
                    help='path to folder saving patch')
parser.add_argument('--max_samples', type=int, default=1e10)
parser.add_argument('--patch_size', type=int, default=512)


if __name__ == '__main__':
    args = parser.parse_args()
    conch, eval_transform = build_conch()  # 用户自行定义构建方式
    pe_loss_fn = PELoss(conch.cuda(), eval_transform)  # 转移到 GPU（如有需要）

    save_dir = args.save_path
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir+"/hr", exist_ok=True)
    os.makedirs(save_dir + "/lr", exist_ok=True)

    wsi_files = get_wsi_files(args.path)
    # 聚类采样
    for wsi_file in wsi_files:
        wsi = WholeSlideImage(wsi_file)
        wsi_name = wsi.name
        coords = getCoords(args.seg_path, wsi_name)
        getPatchFromWsiRoi_PeCluster(wsi, coords, pe_loss_fn, save_dir, ps=args.patch_size, max_samples=args.max_samples)
    # 随机采样
    # for wsi_file in wsi_files:
    #     wsi = WholeSlideImage(wsi_file)
    #     wsi_name = wsi.name
    #     coords = getCoords(args.seg_path, wsi_name)
    #     getPatchFromWsiRoi(wsi, coords, save_dir, ncpu=16, max_samples=args.max_samples)


