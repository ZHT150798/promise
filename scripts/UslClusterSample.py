import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import igraph as ig
import leidenalg
from sklearn.cluster import KMeans
from tqdm import tqdm  # 导入 tqdm 用于进度监控


def image_leiden_clustering(patches, n_clusters, output_path, k_neighbors=10, resolution=1.0):
    """
    对给定的图像 patches 进行 Leiden 聚类，将它们分为 n_clusters 类，
    并将不同类别的图像保存到以类别命名的文件夹中，同时输出每个类别的图像数量。

    参数:
      patches: list，包含每个 patch 的图像（如 numpy 数组）
      n_clusters: int，目标类别数
      output_path: str，保存聚类结果的根目录
      k_neighbors: int，每个 patch 选取的最近邻个数（默认 10）
      resolution: float，Leiden 算法的分辨率参数（默认 1.0）
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 1. 特征提取：这里采用简单的扁平化像素作为特征（实际场景可替换为 CNN 特征）
    features = []
    for patch in tqdm(patches, desc="提取特征"):
        features.append(patch.flatten())
    features = np.array(features)

    # 2. 构建相似度图：计算余弦相似度
    sim_matrix = cosine_similarity(features)
    num_patches = features.shape[0]
    edges = []
    weights = []
    # 利用 tqdm 监控每个 patch 构造相似度图的进度
    for i in tqdm(range(num_patches), desc="构建相似度图"):
        neighbor_indices = np.argsort(-sim_matrix[i])[1:k_neighbors + 1]
        for j in neighbor_indices:
            edges.append((i, j))
            weights.append(sim_matrix[i][j])

    # 使用 igraph 构建图结构
    g = ig.Graph()
    g.add_vertices(num_patches)
    g.add_edges(edges)
    g.es['weight'] = weights

    # 3. 使用 Leiden 算法进行聚类
    partition = leidenalg.find_partition(g,
                                         leidenalg.RBConfigurationVertexPartition,
                                         weights='weight',
                                         resolution_parameter=resolution)
    clusters = partition.membership

    # # 如果 Leiden 得到的类别数不等于目标 n_clusters，则采用 KMeans 进行二次划分
    # unique_clusters = set(clusters)
    # if len(unique_clusters) != n_clusters:
    #     print(f"Leiden 得到的类别数为 {len(unique_clusters)}，与目标 {n_clusters} 不符，采用 KMeans 进行二次划分...")
    #     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    #     clusters = kmeans.fit_predict(features)

    # 4. 保存每个类别的图像，并统计每个类别的数量
    cluster_counts = {}
    for idx, cluster in tqdm(enumerate(clusters), total=len(clusters), desc="保存聚类结果"):
        cluster_dir = os.path.join(output_path, f"class_{cluster}")
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)
        # 保存图像，文件名格式为 patch_{idx}.png
        image_path = os.path.join(cluster_dir, f"patch_{idx}.png")
        cv2.imwrite(image_path, patches[idx])
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

    # 输出每个类别的图像数量
    for cluster, count in cluster_counts.items():
        print(f"类别 class_{cluster}：{count} 张图像")

    return clusters, cluster_counts


def load_patches_from_folder(target_path):
    """
    从目标路径中读取所有图片文件，并返回包含图像数据的列表。
    这里只读取常见图像格式（如 .png, .jpg）。
    """
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    patches = []
    image_files = []
    for file in os.listdir(target_path):
        ext = os.path.splitext(file)[1].lower()
        if ext in valid_extensions:
            image_files.append(os.path.join(target_path, file))

    image_files.sort()  # 按文件名排序

    # 利用 tqdm 监控图像加载进度
    for file in tqdm(image_files, desc="加载图像"):
        img = cv2.imread(file)
        if img is not None:
            patches.append(img)
    return patches


if __name__ == '__main__':
    # 指定目标路径（存放 patch 图像的文件夹）和输出路径
    target_path = "./patch/hr"  # 请替换为实际的输入路径
    output_path = "./patch/clusters"  # 请替换为实际的输出路径

    # 从目标路径读取所有图像
    patches = load_patches_from_folder(target_path)

    # 如果没有读取到图像，则提示错误
    if not patches:
        print("目标路径下未找到有效图像，请检查路径或图像格式。")
    else:
        # 设置目标聚类数，例如 3 类
        n_clusters = 3
        clusters, counts = image_leiden_clustering(patches, n_clusters, output_path, k_neighbors=10, resolution=1.5)

