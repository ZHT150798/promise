import os
import shutil

# 文件夹路径
folder_a = '/media/dell/data/zhangv1/Datasets/downstream/shandong/'
folder_b = '/media/dell/data/zhangv1/Datasets/downstream/shandong/sr'
folder_c = '/media/dell/data/zhangv1/Datasets/downstream/shandong/wrong'  # 目标路径，如果不存在将自动创建

# 允许的图像扩展名（统一小写）
image_exts = {'.svs', '.tiff', '.png', '.jpg'}

# 获取 A 和 B 中图像文件的文件名（不含扩展名）
a_files = [f for f in os.listdir(folder_a)
           if os.path.isfile(os.path.join(folder_a, f)) and os.path.splitext(f)[1].lower() in image_exts]
b_files = [f for f in os.listdir(folder_b)
           if os.path.isfile(os.path.join(folder_b, f)) and os.path.splitext(f)[1].lower() in image_exts]

a_basenames = set(os.path.splitext(f)[0] for f in a_files)
b_basenames = set(os.path.splitext(f)[0] for f in b_files)

# 找出只在 A 中存在的图像文件（按不含扩展名对比）
missing_basenames = a_basenames - b_basenames

# 创建目标文件夹（如不存在）
os.makedirs(folder_c, exist_ok=True)

# 移动对应文件
for f in a_files:
    base, ext = os.path.splitext(f)
    if base in missing_basenames:
        src = os.path.join(folder_a, f)
        dst = os.path.join(folder_c, f)
        shutil.move(src, dst)
        print(f"Moved: {f} → {folder_c}")
