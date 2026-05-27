import os

# 设置数据集的目录路径
dataset_dir = "/media/dell/data/zhangv1/Datasets/TMA/TMA_512/valid_500/"

# 获取目录下所有的图片文件，假设是以 .png 结尾
image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]

# 获取图片数量
num_images = len(image_files)

# 打开并写入文件
with open("TMA_test.txt", "w") as f:
    for image_file in image_files:
        # 写入完整路径
        f.write(os.path.join(dataset_dir, image_file) + "\n")

print(f"文件路径已写入 dataset_paths.txt，共 {num_images} 张图片")
