#!/usr/bin/env python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
import torch
import argparse

# 导入评估函数（确保该模块在 PYTHONPATH 或与当前脚本在同一目录下）
from my_utils.metrics import eval_slide


def parse_args():
    parser = argparse.ArgumentParser(description="WSI Level Evaluation")
    parser.add_argument('--hr_folder', type=str, default="/media/dell/data/zhangv1/WSISR/WSI_Level_SR/wsisr_pipeline/preset/datasets/tcga_l/test/slide/hr")
    parser.add_argument('--sr_folder', type=str, default="/media/dell/data/zhangv1/WSISR/WSI_Level_SR/wsisr_pipeline/preset/datasets/tcga_l/test/slide/sr_v4_gan_align")
    parser.add_argument('--coord_folder', type=str, default="/media/dell/data/zhangv1/WSISR/WSI_Level_SR/wsisr_pipeline/preset/datasets/tcga_l/test/seg_1024/patches")
    parser.add_argument('--mag', type=str, default="40x", help="image magnification", choices=["40x", "20x"])
    parser.add_argument('--log_file', type=str, default=None, help="Evaluation log save path, default in sr_folder")
    return parser.parse_args()


def main():
    args = parse_args()

    # 如果未指定日志路径，则默认在 sr_folder 下创建
    if args.log_file is None:
        args.log_file = os.path.join(args.sr_folder, 'wsi_level_evaluation_log.txt')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 调用评估函数
    results = eval_slide(args.hr_folder, args.sr_folder, args.coord_folder, mag=args.mag,
                         log_file=args.log_file, device=device)

    print(json.dumps(results, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    main()
