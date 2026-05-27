import os, sys

import cv2
import torch.nn as nn
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import igraph as ig
import leidenalg
from tqdm import tqdm
sys.path.append("/media/dell/data/zhangv1/WSISR/WSI_Level_SR/wsisr_pipeline")
from my_utils.fd_loss.conch import build_conch


# 假设 build_conch() 返回 conch 模型和 eval_transform（均已初始化）
# def build_conch():
#     ...
#     return conch, eval_transform

class PELoss(nn.Module):
    def __init__(self, conch, eval_transform):
        super(PELoss, self).__init__()
        self.conch = conch
        self.eval_transform = eval_transform
        # 固定 conch 模型参数，不进行梯度更新
        self.conch.requires_grad_(False)

    def forward(self, sr, hr, pad_size=1024):
        """
        原始 forward 方法计算两个 batch 之间的 L2 损失（略）。
        """
        # 此处略去 forward 方法实现，见之前示例
        pass

    def extract_feature(self, img_tensor, pad_size=1024):
        """
        输入单张图像 tensor (形状为 (C, H, W))，经过 padding、resize、eval_transform 和 conch 映射
        返回 768 维特征向量。
        """
        # 添加 batch 维度
        img_tensor = img_tensor.unsqueeze(0)  # (1, C, H, W)
        _, _, H, _ = img_tensor.shape
        if H != 1024:
            pad = max(pad_size - H, 0)
            pad_top = pad // 2
            pad_bottom = pad - pad_top
            img_tensor = F.pad(img_tensor, (pad_top, pad_bottom, pad_top, pad_bottom), mode='constant', value=0)
        # resize 到 (448, 448)
        img_tensor = F.interpolate(img_tensor, size=(448, 448), mode='bilinear')
        # 使用 eval_transform 进行预处理（例如归一化、resize等），注意 eval_transform 要接收 tensor
        transformed = self.eval_transform(img_tensor)
        # 通过 conch 模型得到特征，去除 batch 维度
        feat = self.conch(transformed.cuda()).squeeze(0)  # 预期输出为 (768,) 或 (1,768) 后 squeeze
        return feat.cpu()

def load_images_as_tensor(target_path):
    """
    从目标路径中读取所有图像文件，转换为 tensor 列表。
    使用 OpenCV 读取图片后转换为 RGB，并利用 transforms.ToTensor 转换为 tensor。
    """
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    image_files = []
    for file in os.listdir(target_path):
        ext = os.path.splitext(file)[1].lower()
        if ext in valid_extensions:
            image_files.append(os.path.join(target_path, file))
    image_files.sort()

    to_tensor = transforms.ToTensor()  # 自动将像素值归一化到 [0,1]
    img_tensors = []
    for file in tqdm(image_files, desc="加载图像并转换为 Tensor"):
        img = cv2.imread(file)
        if img is None:
            continue
        # OpenCV 读取的是 BGR，转换为 RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = to_tensor(img)
        img_tensors.append((file, tensor))
    return image_files, img_tensors


def feature_clustering(img_tensors, pe_loss_fn, n_clusters, output_path, k_neighbors=10, resolution=1.0):
    """
    利用 pe_loss_fn 提取图像特征（768 维），并基于余弦相似度构建图进行 Leiden 聚类，
    最后将聚类结果保存到 output_path 下各类别文件夹中，并输出每个类别图像数量。
    """
    # 1. 特征提取
    features = []
    for _, tensor in tqdm(img_tensors, desc="提取图像特征"):
        feat = pe_loss_fn.extract_feature(tensor)  # 得到 768 维特征 (tensor)
        features.append(feat.cpu().detach().numpy())
    features = np.array(features)

    # 2. 构建相似度图（余弦相似度）
    sim_matrix = cosine_similarity(features)
    num_imgs = features.shape[0]
    edges = []
    weights = []
    for i in tqdm(range(num_imgs), desc="构建相似度图"):
        neighbor_indices = np.argsort(-sim_matrix[i])[1:k_neighbors + 1]  # 排除自身
        for j in neighbor_indices:
            edges.append((i, j))
            weights.append(sim_matrix[i][j])

    g = ig.Graph()
    g.add_vertices(num_imgs)
    g.add_edges(edges)
    g.es['weight'] = weights

    # 3. Leiden 聚类
    partition = leidenalg.find_partition(g,
                                         leidenalg.RBConfigurationVertexPartition,
                                         weights='weight',
                                         resolution_parameter=resolution)
    clusters = partition.membership

    # # 如果 Leiden 得到的类别数不等于目标 n_clusters，则采用 KMeans 二次划分
    # unique_clusters = set(clusters)
    # if len(unique_clusters) != n_clusters:
    #     print(f"Leiden 得到的类别数为 {len(unique_clusters)}，与目标 {n_clusters} 不符，采用 KMeans 进行二次划分...")
    #     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    #     clusters = kmeans.fit_predict(features)

    # 4. 保存聚类结果
    cluster_counts = {}
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for idx, (file, _) in tqdm(enumerate(img_tensors), total=len(img_tensors), desc="保存聚类结果"):
        cluster = clusters[idx]
        cluster_dir = os.path.join(output_path, f"class_{cluster}")
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)
        # 直接复制原图到对应类别目录中
        filename = os.path.basename(file)
        dst_path = os.path.join(cluster_dir, filename)
        # 如果需要保存处理后的图像（例如 RGB 格式），也可以采用 cv2.imwrite，此处直接复制源文件
        cv2.imwrite(dst_path, cv2.imread(file))
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

    for cluster, count in cluster_counts.items():
        print(f"类别 class_{cluster}：{count} 张图像")

    return clusters, cluster_counts


if __name__ == "__main__":
    # 初始化 conch 模型和 eval_transform（请确保 build_conch 已经实现）
    conch, eval_transform = build_conch()  # 用户自行定义构建方式
    pe_loss_fn = PELoss(conch.cuda(), eval_transform)  # 转移到 GPU（如有需要）

    # 指定图像所在的目标路径与结果保存路径
    target_path = "./patch/hr"  # 请替换为实际的输入路径
    output_path = "./patch/clusters_pe_k7_r1.0"  # 请替换为实际的输出路径

    # 加载图像并转换为 tensor
    img_files, img_tensors = load_images_as_tensor(target_path)
    if not img_tensors:
        print("目标路径下未找到有效图像，请检查路径或图像格式。")
    else:
        # 设置目标聚类数，例如 3 类
        n_clusters = 3
        clusters, counts = feature_clustering(img_tensors, pe_loss_fn, n_clusters, output_path, k_neighbors=7, resolution=1.0)


