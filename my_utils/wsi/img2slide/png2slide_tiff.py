import pyvips
import os

def convert_to_pyramidal_tiff(input_path, output_path=None):
    """
    将普通图像 (PNG/JPEG/TIFF) 转换为 pyramidal TIFF，适合在 QuPath 中浏览

    Args:
        input_path (str): 输入图像路径
        output_path (str, optional): 输出路径 (默认与输入同名 .tif)
    """
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + "_pyramidal.tif"

    # 读取输入图像
    image = pyvips.Image.new_from_file(input_path, access="sequential")

    # 保存为 pyramidal tiff
    image.tiffsave(
        output_path,
        tile=True,               # 开启切片存储
        pyramid=True,            # 启用多层金字塔
        compression="jpeg",      # 压缩方式（可改为 "lzw" 或 "none"）
        tile_width=512,
        tile_height=512,
        bigtiff=True             # 支持大文件（>4GB）
    )

    print(f"✅ 转换完成: {output_path}")


if __name__ == "__main__":
    # 在这里直接修改路径
    input_file = r"test_png2tiff/TCGA-2Y-A9GU-01Z-00-DX1.700CBBD7-9F58-470D-A85B-C7BAE5608018.png"       # 输入图像路径
    output_file = r"test_png2tiff/TCGA-2Y-A9GU-01Z-00-DX1.700CBBD7-9F58-470D-A85B-C7BAE5608018.tif"  # 输出图像路径（可省略，默认自动生成）

    convert_to_pyramidal_tiff(input_file, output_file)
