#!/usr/bin/env python3
import os
import sys
import openslide


def get_scanner_model(wsi_path):
    """
    尝试从WSI图像中提取扫描仪型号信息，通常存储在 'openslide.vendor' 属性中。
    若未找到该属性或发生错误，则返回 'Unknown' 或打印错误信息。
    """
    try:
        slide = openslide.OpenSlide(wsi_path)
        properties = slide.properties
        #print(properties)
        # 检查 'openslide.vendor' 属性，部分文件可能还包含 'openslide.comment' 等其他信息
        scanner_model = properties.get('openslide.vendor', 'Unknown')
        return scanner_model
    except Exception as e:
        print(f"Error reading WSI file '{wsi_path}': {e}")
        return None


def find_wsi_files(folder):
    """
    遍历给定文件夹，查找所有扩展名为 .svs 或 .npdi 的文件（不区分大小写）。
    返回一个包含所有符合条件文件完整路径的列表。
    """
    matches = []
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.lower().endswith('.svs') or filename.lower().endswith('.npdi'):
                full_path = os.path.join(root, filename)
                matches.append(full_path)
    return matches


if __name__ == '__main__':
    folder_path = "/media/dell/data/zhangv1/WSISR/WSI_Level_SR/wsisr_pipeline/preset/datasets/tcga/train/wsi"

    wsi_files = find_wsi_files(folder_path)
    if not wsi_files:
        print(f"No .svs or .npdi files found in '{folder_path}'.")
    else:
        print(f"Found {len(wsi_files)} WSI files:")
        for file_path in wsi_files:
            model = get_scanner_model(file_path)
            if model is not None:
                print(f"File: {file_path}\nScanner model: {model}\n")
            else:
                print(f"File: {file_path}\nScanner model: Not available\n")

