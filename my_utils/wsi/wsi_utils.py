# coding=utf-8
import openslide
import numpy as np
import scipy.misc
# import cv2
import numpy as np
import os
import PIL.Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import time
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyvips
import h5py
from multiprocessing import Pool, Manager
from itertools import repeat
import sys
import random
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import igraph as ig
import leidenalg


def readWSIByMag(path, level):
    pyvips.Image.new_from_file(path, level=level)

def allocate_samples(cluster_map, total_patches, max_samples):
    """
    根据每个类别在总体中所占比例，采用最大余数法分配采样数，
    确保总采样数等于 max_samples，并且每个类别采样数至少为1，
    且不超过该类别的总样本数。

    参数:
      cluster_map: dict, 键为类别号，值为该类别中样本索引列表
      total_patches: 所有样本总数
      max_samples: 目标采样总数

    返回:
      cluster_alloc: dict, 键为类别号，值为分配的采样数
    """
    cluster_alloc = {}  # 初步分配（取整）结果
    remainder_map = {}  # 每个类别的余数
    for cluster, indices in cluster_map.items():
        exact = (len(indices) / total_patches) * max_samples
        # 初步分配：至少 1，且不能超过该类别样本总数
        allocated = max(1, int(exact))
        allocated = min(allocated, len(indices))
        cluster_alloc[cluster] = allocated
        remainder_map[cluster] = exact - allocated

    current_total = sum(cluster_alloc.values())
    diff = max_samples - current_total

    if diff > 0:
        # 采样数不足，从余数最大的类别中补充
        # 排序时：余数越大，优先补充
        for cluster in sorted(cluster_map.keys(), key=lambda c: remainder_map[c], reverse=True):
            available = len(cluster_map[cluster]) - cluster_alloc[cluster]
            # 能补充的数目不能超过该类别剩余可选样本
            add = min(diff, available)
            cluster_alloc[cluster] += add
            diff -= add
            if diff == 0:
                break
    elif diff < 0:
        # 采样数过多，从余数最小的类别中减少（但每个类别至少保留1个）
        for cluster in sorted(cluster_map.keys(), key=lambda c: remainder_map[c]):
            removable = cluster_alloc[cluster] - 1
            remove_count = min(-diff, removable)
            cluster_alloc[cluster] -= remove_count
            diff += remove_count
            if diff == 0:
                break
    # 最终检查
    assert sum(cluster_alloc.values()) == max_samples, "分配的采样总数不等于目标"

    return cluster_alloc


def wsi2Patch(slide, coord, save_dir, name, ps=512, save_ext='.png', needlr=True):
    x, y = coord
    # if slide.level_dim[0][0] <= x + ps or slide.level_dim[0][1] <= y + ps:
    #     return None
    if needlr:
        hr, lr = slide.getRegion(x, y, ps, ps, True)
        # if not filter_background(hr, 0.05):
        #     return None
        hr_savepath = f'{save_dir}/hr/{name}-{x}-{y}{save_ext}'
        lr_savepath = f'{save_dir}/lr/{name}-{x}-{y}{save_ext}'
        cv2.imwrite(hr_savepath, cv2.cvtColor(hr, cv2.COLOR_RGBA2BGR))
        cv2.imwrite(lr_savepath, cv2.cvtColor(lr, cv2.COLOR_RGBA2BGR))
        return 1


def getPatchFromWsiRoi(slide, coords, save_dir, ps=512, save_ext='.png', needlr=True, ncpu=8, max_samples=100):
    name = slide.name.split(".")[0]
    if len(coords) < 100:
        print(f"Processing {slide.name}")
    if len(coords) >= max_samples:
        indices = random.sample(range(len(coords)), max_samples)
        coords = [coords[i] for i in indices]
        # coords = coords[:max_samples]

    total_tasks = len(coords)
    with ThreadPoolExecutor() as executor:
        futures = []
        # 使用 tqdm 来监控进度
        progress_bar = tqdm(total=total_tasks, desc="Processing Patches")
        # 提交任务到线程池
        for coord in coords:
            futures.append(executor.submit(wsi2Patch, slide, coord, save_dir, name, ps, save_ext, needlr))

        # 等待任务完成
        for future in as_completed(futures):
            future.result()  # 获取返回值或异常
            result = future.result()  # 获取返回值或异常
            if result is not None:  # 只有当返回值不为 None 时才更新进度
                progress_bar.update(1)
    progress_bar.close()  # 关闭进度条


def wsi2tensor(slide, coords, ps=512):
    patch_tensors, patch_coords = [], []
    to_tensor = transforms.ToTensor()
    for coord in coords:
        x, y = coord
        hr, _ = slide.getRegion(x, y, ps, ps, True)
        tensor = to_tensor(hr)
        patch_tensors.append(tensor)
        patch_coords.append(coord)
    return patch_tensors, patch_coords


def wsi2tensor_parallel(slide, coords, ps=512):
    """
    并行处理 wsi2tensor，确保 patch 和 coords 一一对应
    """

    def load_patch(idx, coord):
        """从 slide 获取单个 patch"""
        x, y = coord
        hr, _ = slide.getRegion(x, y, ps, ps, True)
        return idx, cv2.cvtColor(hr, cv2.COLOR_RGBA2RGB)  # 确保返回索引和对应的 patch

    patches = [None] * len(coords)  # 预分配列表，保持顺序
    with ThreadPoolExecutor() as executor:
        future_to_idx = {executor.submit(load_patch, i, coord): i for i, coord in enumerate(coords)}
        for future in tqdm(as_completed(future_to_idx), total=len(coords), desc="加载 Patch"):
            idx, patch = future.result()
            patches[idx] = patch  # 保持原始顺序不变

    return patches, coords  # 返回的 patch 顺序与 coords 对应


def getPatchFromWsiRoi_PeCluster(slide, coords, pe_loss_fn, save_dir, ps=512, save_ext='.png', needlr=True,
                                 max_samples=100):
    name = slide.name.split(".")[0]
    if len(coords) < max_samples:
        max_samples = len(coords)
    to_tensor = transforms.ToTensor()

    # 1. 提取所有patch的tensor集合
    features = []
    patch, coords = wsi2tensor_parallel(slide, coords)  # wsi2tensor 返回的patch顺序和coords必须保持一致
    for i in tqdm(range(len(patch)), desc="提取图像特征"):
        feat = pe_loss_fn.extract_feature(to_tensor(patch[i]))  # 得到768维特征 (tensor)
        features.append(feat.cpu().detach().numpy())
    features = np.array(features)

    # 2. 构建相似度图（基于余弦相似度）
    k_neighbors, resolution = 7, 1.0
    sim_matrix = cosine_similarity(features)
    num_imgs = features.shape[0]
    edges = []
    weights = []
    for i in tqdm(range(num_imgs), desc="构建相似度图"):
        # 排除自身，选择前 k_neighbors 个最相似的样本
        neighbor_indices = np.argsort(-sim_matrix[i])[1:k_neighbors + 1]
        for j in neighbor_indices:
            edges.append((i, j))
            weights.append(sim_matrix[i][j])

    g = ig.Graph()
    g.add_vertices(num_imgs)
    g.add_edges(edges)
    g.es['weight'] = weights

    # 3. Leiden 聚类
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=resolution
    )
    clusters = partition.membership

    # 建立每个聚类类别到索引列表的映射
    cluster_map = {}
    total_patches = len(coords)
    for idx, cluster in enumerate(clusters):
        cluster_map.setdefault(cluster, []).append(idx)

    # 记录每个类别实际采样的数量，用于统计输出
    cluster_sample_counts = {}
    selected_indices = []
    # 根据 cluster_map、total_patches、max_samples 进行采样数分配
    cluster_alloc = allocate_samples(cluster_map, total_patches, max_samples)
    # 根据每个类别在所有样本中所占比例分配采样数
    for cluster, indices in cluster_map.items():
        allocated = cluster_alloc[cluster]
        # 当 allocated 小于该类别总数时，随机采样，否则全部选取
        if allocated < len(indices):
            chosen = random.sample(indices, allocated)
        else:
            chosen = indices.copy()
        cluster_sample_counts[cluster] = len(chosen)
        selected_indices.extend(chosen)

    # 输出聚类统计信息
    print("聚类的类别总数：", len(cluster_map))
    for cluster, indices in cluster_map.items():
        cluster_ratio = cluster_sample_counts[cluster] / len(indices)
        print(
            f"类别 {cluster} 的抽样比值： {cluster_ratio:.4f}（采样数：{cluster_sample_counts[cluster]}/该类总数：{len(indices)}）")

    # 确保保存目录下的 hr 和 lr 子目录存在
    hr_dir = os.path.join(save_dir, 'hr')
    lr_dir = os.path.join(save_dir, 'lr')
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)

    # 4. 对采样到的每个 coord 保存 patch 图像
    for idx in tqdm(selected_indices, desc="保存采样 Patch"):
        coord = coords[idx]
        x, y = coord
        if needlr:
            hr, lr = slide.getRegion(x, y, ps, ps, True)
            # 注意：getRegion 得到的是 RGBA，转换成 RGB
            hr_rgb = cv2.cvtColor(hr, cv2.COLOR_RGBA2RGB)
            # 使用前景/背景分割评估 patch 质量
            background_ratio = segment_foreground_background(hr_rgb)
            # 设置阈值，背景过多则跳过保存
            if background_ratio > 0.70:
                continue  # 丢弃背景占比太大的 patch
            hr_savepath = os.path.join(hr_dir, f'{name}-{x}-{y}{save_ext}')
            lr_savepath = os.path.join(lr_dir, f'{name}-{x}-{y}{save_ext}')
            cv2.imwrite(hr_savepath, cv2.cvtColor(hr, cv2.COLOR_RGBA2BGR))
            cv2.imwrite(lr_savepath, cv2.cvtColor(lr, cv2.COLOR_RGBA2BGR))


def getCoords(path, name):
    # 打开一个 h5 文件
    with h5py.File(f'{path}/patches/{name.split(".svs")[0]}.h5', 'r') as file:
        coords = file['coords']
        return coords[:]


def get_wsi_files(folder_path, extensions=None):
    """
    获取指定文件夹下的所有 WSI 文件路径。

    Args:
        folder_path (str): 文件夹路径
        extensions (list[str], optional): 支持的扩展名（默认包含常见WSI格式）

    Returns:
        list[str]: 符合条件的WSI文件路径列表
    """
    if extensions is None:
        extensions = ['.svs', '.tif', '.tiff', '.ndpi', '.mrxs', '.scn', '.bif', '.vms', '.vmu']

    wsi_files = []
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in extensions):
            wsi_files.append(os.path.join(folder_path, file))

    return wsi_files


def segment_foreground_background(img_rgb, use_otsu=False, sthresh=20, sthresh_up=255,
                                  mthresh=7, close=5):
    """
    对给定图像进行前景背景分割：
      - 使用 HSV 空间的饱和度通道分割
      - 对分割结果进行形态学闭运算平滑处理（可选）
      - 计算背景像素占比，并将背景置为黑色
      - 可视化原图与前景图

    参数:
      image_path: 图像路径
      use_otsu: 是否采用大津阈值法（True）或固定阈值（False）
      sthresh: 固定阈值下的阈值
      sthresh_up: 固定阈值时二值化后的最大值（通常为 255）
      mthresh: 中值滤波的核大小（必须为奇数）
      close: 形态学闭运算的核大小，若 <=0 则不进行闭运算

    返回:
      background_ratio: 背景像素占比
      foreground_img: 只保留前景区域（背景置为黑色）的 RGB 图像
    """
    if img_rgb is None or not isinstance(img_rgb, np.ndarray):
        raise ValueError("输入图像无效，请检查格式！")

    # 转换到 HSV 颜色空间
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # 提取饱和度通道（S 通道）
    sat = img_hsv[:, :, 1]

    # 中值滤波去噪
    sat_blur = cv2.medianBlur(sat, mthresh)

    # 阈值分割
    if use_otsu:
        _, binary = cv2.threshold(sat_blur, 0, sthresh_up, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(sat_blur, sthresh, sthresh_up, cv2.THRESH_BINARY)


    # 形态学闭运算：填充前景内部的小孔（若 close > 0）
    if close > 0:
        kernel = np.ones((close, close), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 计算背景像素占比（这里背景设为 0，前景为 255）
    total_pixels = binary.size
    background_pixels = np.sum(binary == 0)
    background_ratio = background_pixels / total_pixels
    del img_rgb, img_hsv, sat, sat_blur, binary, kernel


    return background_ratio #, foreground_img


# 示例调用
# background_ratio, fg_img = segment_foreground_background("path/to/your/image.png")
if __name__ == "__main__":
    image_path ="/media/dell/data/zhangv1/WSISR/WSI_Level_SR/wsisr_pipeline/preset/datasets/tcga_v2/test/patch/hr"
    use_otsu = False
    valid_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    image_files = [os.path.join(image_path, f) for f in os.listdir(image_path)
                   if os.path.splitext(f)[1].lower() in valid_exts]
    for img_path in image_files:
        print("Processing:", img_path)
        try:
            background_ratio, fg_img = segment_foreground_background(
                img_path, use_otsu=use_otsu)
        except Exception as e:
            print("Error processing {}: {}".format(img_path, e))
            continue