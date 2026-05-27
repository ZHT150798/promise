import os
import openslide
from tqdm import tqdm

def count_slides_with_downsampling(slide_dir, target_downsampling, tolerance=0.01):
    """
    统计 slide_dir 目录下具有指定下采样因子的切片数量。

    参数:
    - slide_dir: str，包含切片图像的文件夹路径
    - target_downsampling: float，要查询的下采样因子（如 4.0, 8.0）
    - tolerance: float，容许误差（默认为 0.01）

    返回:
    - 匹配数量
    """
    count = 0
    valid_files = []

    # 获取所有 slide 文件
    slide_files = [f for f in os.listdir(slide_dir)
                   if f.lower().endswith(('.svs', '.tiff', '.ndpi', '.mrxs', '.scn'))]

    for fname in tqdm(slide_files, desc="Processing slides"):
        slide_path = os.path.join(slide_dir, fname)
        try:
            slide = openslide.OpenSlide(slide_path)
            downsamples = slide.level_downsamples  # List[float]

            # 检查是否存在一个 level 的下采样值接近目标值
            if any(abs(ds - target_downsampling) < tolerance for ds in downsamples):
                count += 1
            else:
                valid_files.append(fname)

        except Exception as e:
            print(f"⚠️ 无法读取文件 {fname}：{e}")

    print(f"\n总切片数量: {len(slide_files)}")
    print(f"包含下采样因子 ≈ {target_downsampling} 的切片数量: {count}")

    return count, valid_files

# 示例调用
if __name__ == "__main__":
    slide_folder = "/media/dell/data3/CPTAC/external_testset/wsi"  # 替换为你的实际文件夹路径
    target_factor = 4.0     # 目标下采样因子
    count, files = count_slides_with_downsampling(slide_folder, target_factor)

    # # 可选：输出匹配的文件名
    # print("\n不匹配的切片文件：")
    # for fname in files:
    #     print(f"  - {fname}")


