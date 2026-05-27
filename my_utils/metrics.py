import os
import sys
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
import h5py
import json
import gc
from glob import glob

sys.path.append(os.getcwd())

import torch
from torchvision import transforms

from PIL import Image
from tqdm import tqdm
import numpy as np
import math
from skimage.util import img_as_float
import torch.nn as nn
import pytorch_fid.fid_score as fid_score
from my_utils.fd_loss.conch import build_conch
from my_utils.fd_loss import pe_loss
import lpips
from transformers import AutoModel
import torchvision.transforms as T

tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
eval_transform = T.Compose([
    T.Resize(448, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(448),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

import os
import shutil


def create_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def psnr(path_input, path_ref, std=False):
    MSE = nn.MSELoss()
    imgs_input = sorted(glob(os.path.join(path_input, '*.jpg')) + glob(os.path.join(path_input, '*.png')))
    psnr_list = []

    for i in tqdm(range(len(imgs_input)), desc="Calculate PSNR", unit="image", position=0, leave=False):
        imgi_name = os.path.basename(imgs_input[i])
        imgr_name = os.path.splitext(imgi_name)[0]+".png"  # 假设参考图像文件名一致

        # 加载并转 float32
        img_input = torch.from_numpy(img_as_float(Image.open(os.path.join(path_input, imgi_name)).convert("RGB"))).float()
        img_ref = torch.from_numpy(img_as_float(Image.open(os.path.join(path_ref, imgr_name)).convert("RGB"))).float()

        # 增加 batch 维度
        img_input = img_input.unsqueeze(0)
        img_ref = img_ref.unsqueeze(0)

        # 计算 PSNR
        mse = MSE(img_input, img_ref)
        psnr_value = 10 * math.log10(1 / mse.item())
        psnr_list.append(psnr_value)

    # 平均值与标准差
    ave_psnr = np.mean(psnr_list)
    std_psnr = np.std(psnr_list)
    if std:
        return ave_psnr, std_psnr
    return ave_psnr




def lpips_score(path_input, path_ref, lpips_model, device, std=False):
    imgs_input = sorted(glob(os.path.join(path_input, '*.jpg')) + glob(os.path.join(path_input, '*.png')))
    lpips_scores = []

    for i in tqdm(range(len(imgs_input)), desc="Calculate LPIPS", unit="image", position=0, leave=False):
        imgi_name = os.path.basename(imgs_input[i])
        imgr_name = os.path.splitext(imgi_name)[0]+".png" # 假设名称一一对应

        # 加载图像并转换为 tensor
        img_input = Image.open(os.path.join(path_input, imgi_name)).convert("RGB")
        img_ref = Image.open(os.path.join(path_ref, imgr_name)).convert("RGB")

        img_input = tensor_transforms(img_input).unsqueeze(0).to(device)
        img_ref = tensor_transforms(img_ref).unsqueeze(0).to(device)

        # 计算 LPIPS 分数
        lpips_value = lpips_model(img_input, img_ref)
        lpips_scores.append(lpips_value.item())

    mean_lpips = np.mean(lpips_scores)
    std_lpips = np.std(lpips_scores)
    if std:
        return mean_lpips, std_lpips
    return mean_lpips


def pe_score(path_input, path_ref, pe_model, device, std=False):
    imgs_input = sorted(glob(os.path.join(path_input, '*.jpg')) + glob(os.path.join(path_input, '*.png')))

    pe_scores = []
    pe_feat_hr, pe_feat_sr = [], []

    for i in tqdm(range(len(imgs_input)), desc="Calculate PE", unit="image", position=0, leave=False):
        imgi_name = os.path.basename(imgs_input[i])
        imgr_name = os.path.splitext(imgi_name)[0]+".png"  # 假设文件名一一对应

        img_input = Image.open(os.path.join(path_input, imgi_name)).convert("RGB")
        img_ref = Image.open(os.path.join(path_ref, imgr_name)).convert("RGB")

        img_input = eval_transform(img_input).unsqueeze(0).to(device)
        img_ref = eval_transform(img_ref).unsqueeze(0).to(device)

        pe_value, sr_feat, hr_feat = pe_model.compute_feat_diff(img_ref, img_input)

        pe_feat_hr.append(hr_feat)
        pe_feat_sr.append(sr_feat)
        pe_scores.append(pe_value.item())

    mean_pe = np.mean(pe_scores)
    std_pe = np.std(pe_scores)
    if std:
        return mean_pe, std_pe
    return mean_pe, torch.stack(pe_feat_hr, dim=0), torch.stack(pe_feat_sr, dim=0)


def pe_score_slide_level(path_input, path_ref, pe_model, device):
    imgs_input = glob(os.path.join(path_input, '*.jpg')) + glob(os.path.join(path_input, '*.png'))
    imgs_input.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    # imgs_ref = glob(os.path.join(path_ref, '*.jpg')) + glob(os.path.join(path_ref, '*.png'))
    ave_pe = 0.0
    pe_feat_hr, pe_feat_sr = [], []
    # 循环计算每对图像的LPIPS
    for i in tqdm(range(len(imgs_input)), desc="Calculate PE", unit="image", position=0, leave=False):
        # 获取输入和参考图像名称
        imgi_name, imgr_name = imgs_input[i].split("/")[-1], imgs_input[i].split("/")[-1]

        # 读取图像并进行预处理
        img_input = Image.open(os.path.join(path_input, imgi_name)).convert("RGB")
        img_ref = Image.open(os.path.join(path_ref, imgr_name)).convert("RGB")

        img_input = eval_transform(img_input).unsqueeze(0).to(device)  # 增加batch维度并迁移到设备
        img_ref = eval_transform(img_ref).unsqueeze(0).to(device)

        pe_value, sr_feat, hr_feat = pe_model.compute_feat_diff(img_ref, img_input)
        pe_feat_hr.append(hr_feat)
        pe_feat_sr.append(sr_feat)
        ave_pe += pe_value.item()

    # 计算平均LPIPS
    ave_pe /= len(imgs_input)

    return ave_pe, torch.stack(pe_feat_hr, dim=0), torch.stack(pe_feat_sr, dim=0)


def compute_se_metric(feature_hr, feature_sr, coords, model, device, patch_size=1024):
    # 扩展 batch 维度： (N, D) -> (1, N, D)
    features_hr = feature_hr.unsqueeze(0)
    features_sr = feature_sr.unsqueeze(0)
    coords = coords.unsqueeze(0).to(torch.int64)

    # features_hr, features_sr = features_hr.to(device), features_sr.to(device)
    coords = coords.to(device)

    with torch.autocast(device.type, torch.float16), torch.inference_mode():
        slide_embedding_hr = model.encode_slide_from_patch_features(features_hr, coords, patch_size)
        slide_embedding_sr = model.encode_slide_from_patch_features(features_sr, coords, patch_size)
        mae = nn.functional.l1_loss(slide_embedding_hr.cpu(), slide_embedding_sr.cpu(), reduction='mean')

    se_metric = mae.item()
    return se_metric


def get_patch_coords_from_h5(h5_path):
    with h5py.File(h5_path, 'r') as f:
        coords = torch.from_numpy(f['coords'][:]).to(torch.int64)
    return coords


def extract_patches(image, coords, patch_size=512):
    patches = []
    for (x, y) in coords:
        patch = image[y:y + patch_size, x:x + patch_size]
        patches.append(patch)
    return patches


def eval_slide(gt_path, sr_path, coord_path, mag="40x", log_file='evaluation_log.txt', device=None):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    titan_model = AutoModel.from_pretrained('/media/dell/data/zhangv1/WSISR/huggingface/TITAN', trust_remote_code=True)
    titan_model = titan_model.to(device)
    conch, _ = build_conch()
    conch.requires_grad_(False)
    pe_loss_fn = pe_loss.PELoss(conch).to(device)
    lpips_loss = lpips.LPIPS(net='vgg').to(device)

    # 获取 hr 和 sr 文件夹下所有 slide 路径（假设所有文件均为图片）
    hr_slide_paths = sorted(glob(os.path.join(gt_path, '*')))
    sr_slide_paths = sorted(glob(os.path.join(sr_path, '*')))

    # 建立 slide 名称到路径字典（假设 hr 与 sr slide 名称匹配）
    hr_dict = {os.path.basename(p): p for p in hr_slide_paths}
    sr_dict = {os.path.basename(p): p for p in sr_slide_paths}

    slide_results = {}
    all_fid, all_psnr, all_lpips, all_pe, all_se = [], [], [], [], []

    print(f"total {len(hr_dict)} slides，start evaluation...")

    for slide_name, hr_path in tqdm(hr_dict.items(), desc="Processing Slides", unit="slide", position=0, colour='green', leave=True):
        if slide_name not in sr_dict:
            print(f"Warning: {slide_name} 在 SR 文件夹中未找到，跳过")
            continue

        sr_path = sr_dict[slide_name]
        hr_slide = cv2.imread(hr_path)
        sr_slide = cv2.imread(sr_path)
        if hr_slide is None or sr_slide is None:
            print(f"Error: 无法读取 {slide_name} 的图片")
            continue
        # 获取 HR 的尺寸
        hr_height, hr_width = hr_slide.shape[:2]
        # 如果 SR 尺寸与 HR 不一致，则进行 resize
        if sr_slide.shape[:2] != (hr_height, hr_width):
            print(f"图像{slide_name} 存在size不匹配，通过bicubic采样对齐")
            sr_slide = cv2.resize(sr_slide, (hr_width, hr_height), interpolation=cv2.INTER_CUBIC)

        # ---------------------------
        # patch level
        # ---------------------------
        base, _ = os.path.splitext(hr_path)
        h5_path = os.path.join(coord_path, base.split("/")[-1]) + ".h5"
        coords_tensor = get_patch_coords_from_h5(h5_path)
        # 转为 list 格式，用于 patch 提取
        coords_list = coords_tensor.cpu().numpy().tolist()


        # 提取 hr 与 sr 的 patch（根据相同coord）
        patch_size = 512 if mag == "20x" else 1024
        hr_patches = extract_patches(hr_slide, coords_list, patch_size=patch_size)
        sr_patches = extract_patches(sr_slide, coords_list, patch_size=patch_size)
        del hr_slide, sr_slide

        if len(hr_patches) == 0 or len(sr_patches) == 0:
            print(f"Warning: {slide_name} no valid patch，skip")
            continue

        num_patches = min(len(hr_patches), len(sr_patches))
        print(f"{slide_name} processing，a total of  {num_patches} patches...")
        # 定义临时文件夹路径
        parent_dir = os.path.dirname(gt_path)
        temp_hr = os.path.join(parent_dir, 'temp_patch_hr')
        temp_sr = os.path.join(parent_dir, 'temp_patch_sr')
        create_folder(temp_hr)
        create_folder(temp_sr)

        for idx in tqdm(range(num_patches), desc=f"Saving patches for {slide_name}", unit="patch", position=1, leave=False):
            hr_patch = hr_patches[idx]
            sr_patch = sr_patches[idx]
            hr_patch_path = os.path.join(temp_hr, f'{idx:05d}.png')
            sr_patch_path = os.path.join(temp_sr, f'{idx:05d}.png')
            cv2.imwrite(hr_patch_path, hr_patch)
            cv2.imwrite(sr_patch_path, sr_patch)
        print(f"Calculate the metrics of the {slide_name}...")
        del hr_patches, sr_patches
        # 使用批量计算函数计算指标
        fid_val = fid_score.calculate_fid_given_paths([temp_hr, temp_sr],
                                                      batch_size=100, device=device,
                                                      dims=2048,
                                                      num_workers=4)
        psnr_val = psnr(temp_hr, temp_sr)
        lpips_val = lpips_score(temp_hr, temp_sr, lpips_loss, device)
        pe_val, pe_feat_hr, pe_feat_sr = pe_score_slide_level(temp_hr, temp_sr, pe_loss_fn, device)

        # 删除临时文件夹
        shutil.rmtree(temp_hr)
        shutil.rmtree(temp_sr)

        # ---------------------------
        # 计算 se 指标
        # ---------------------------
        # demo_h5_path = "/media/dell/data/zhangv1/WSISR/WSI_Level_SR/wsisr_pipeline/preset/datasets/tcga/test/seg_s/features/h5_files/TCGA-55-A491-01Z-00-DX1.E5F3B4E5-18EA-4067-AE28-119DABCE739A.h5"
        # file = h5py.File(demo_h5_path, 'r')
        # features = torch.from_numpy(file['features'][:]).to(device)
        se_metric = compute_se_metric(pe_feat_hr, pe_feat_sr, coords_tensor, titan_model, device,
                                      patch_size=patch_size)
        del pe_feat_hr, pe_feat_sr

        slide_metrics = {
            'fid_mean': fid_val,
            'psnr_mean': psnr_val,
            'lpips_mean': lpips_val,
            'pe_mean': pe_val,
            'se_metric': se_metric,
            'num_patches': num_patches
        }
        slide_results[slide_name] = slide_metrics

        all_fid.append(slide_metrics['fid_mean'])
        all_psnr.append(slide_metrics['psnr_mean'])
        all_lpips.append(slide_metrics['lpips_mean'])
        all_pe.append(slide_metrics['pe_mean'])
        if se_metric is not None:
            all_se.append(se_metric)
        gc.collect()
        torch.cuda.empty_cache()

    overall_metrics = {
        'fid_mean': np.mean(all_fid) if all_fid else None,
        'psnr_mean': np.mean(all_psnr) if all_psnr else None,
        'lpips_mean': np.mean(all_lpips) if all_lpips else None,
        'pe_mean': np.mean(all_pe) if all_pe else None,
        'se_metric_mean': np.mean(all_se) if all_se else None,
        'fid_std': np.std(all_fid) if all_fid else None,
        'psnr_std': np.std(all_psnr) if all_psnr else None,
        'lpips_std': np.std(all_lpips) if all_lpips else None,
        'pe_std': np.std(all_pe) if all_pe else None,
        'se_metric_std': np.std(all_se) if all_se else None,
        'num_slides': len(all_fid),
        'patch_size': patch_size,

    }

    final_results = {
        'slide_results': slide_results,
        'overall_metrics': overall_metrics
    }

    # 保存结果日志为 JSON 文件
    print("评估完成，结果保存至:", log_file)
    with open(log_file, 'w') as f:
        json.dump(final_results, f, indent=4)
    return final_results


if __name__ == '__main__':
    # 根据实际情况设置 hr 与 sr 文件夹路径
    hr_folder_path = '/media/dell/data/zhangv1/WSISR/WSI_Level_SR/wsisr_pipeline/preset/datasets/tcga/test/slide_s/hr'
    sr_folder_path = '/media/dell/data/zhangv1/WSISR/WSI_Level_SR/wsisr_pipeline/preset/datasets/tcga/test/slide_s/sr_v2_pe_norm'
    coord_path = '/media/dell/data/zhangv1/WSISR/WSI_Level_SR/wsisr_pipeline/preset/datasets/tcga/test/seg_s/patches'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_path = sr_folder_path + '/wsi_level_evaluation_log.txt'
    results = eval_slide(hr_folder_path, sr_folder_path, coord_path, mag="40x", log_file=log_path,
                         device=device)
    print(json.dumps(results, indent=4, ensure_ascii=False))
