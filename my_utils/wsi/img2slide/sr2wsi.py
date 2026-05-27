from openslide import OpenSlide
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import tifffile as tiff
import pyvips
import os
import time

def sr_to_wsi(original_svs, sr_image_path, output_path, method="PIL", default_mpp=0.25):
    """
    将 SR 图像重建为 WSI 风格的金字塔 TIFF，并比对原始 WSI 信息。

    参数：
        method: "PIL" 或 "pyvips"，选择生成方式
    """
    start_time = time.time()

    # 1. 读取原始 WSI
    slide = OpenSlide(original_svs)
    levels = slide.level_count
    level_dims = slide.level_dimensions  # [(w0,h0), (w1,h1), ...]
    # 尝试读取 mpp
    mpp_x_str = slide.properties.get('openslide.mpp-x')
    mpp_y_str = slide.properties.get('openslide.mpp-y')
    if mpp_x_str is None or mpp_y_str is None:
        print(f"⚠️ 未读取到原 WSI 的 mpp 信息，将使用默认值 {default_mpp} µm/px")
        mpp_x = mpp_y = default_mpp
    else:
        mpp_x = float(mpp_x_str)
        mpp_y = float(mpp_y_str)

    print(f"原 WSI 层数: {levels}, mpp_x: {mpp_x}, mpp_y: {mpp_y}")

    sr_img = pyvips.Image.new_from_file(sr_image_path, access="sequential")
    # if sr_img.width != level_dims[0][0] or sr_img.height != level_dims[0][1]:
    #     scale_x = level_dims[0][0] / sr_img.width
    #     scale_y = level_dims[0][1] / sr_img.height
    #     print(f"⚠️ SR 图像尺寸 ({sr_img.width},{sr_img.height}) 与原 WSI 第一层 {level_dims[0]} 不匹配，自动 resize")
    #     sr_img = sr_img.resize(scale_x, kernel='lanczos')
    #     sr_img = sr_img.resize(scale_y, kernel='lanczos', v=True)  # v=True 表示垂直缩放


    sr_img = sr_img.cast("uchar")

    # 再写入 tiff 文件
    sr_img.tiffsave(
        output_path,
        tile=True,
        pyramid=True,
        compression="jpeg",
        tile_width=256,
        tile_height=256,
        bigtiff=True,
        xres=1000/mpp_x,
        yres=1000/mpp_y,
        Q=75
    )



    end_time = time.time()
    print(f"✅ SR WSI ({method}) 重建完成: {output_path}")
    print(f"生成耗时: {end_time - start_time:.2f} 秒")


# -------------------
# 使用示例
original_svs = r"../test_png2tiff/TCGA-2Y-A9GU-01Z-00-DX1.700CBBD7-9F58-470D-A85B-C7BAE5608018.svs"
sr_image_path = r"../test_png2tiff/TCGA-2Y-A9GU-01Z-00-DX1.700CBBD7-9F58-470D-A85B-C7BAE5608018.png"
# output_wsi = r"../test_png2tiff/TCGA-2Y-A9GU-01Z-00-DX1.700CBBD7-9F58-470D-A85B-C7BAE5608018.tif"
output_wsi_vips = r"../test_png2tiff/TCGA-2Y-A9GU-01Z-00-DX1.700CBBD7-9F58-470D-A85B-C7BAE5608018.tif"

# sr_to_wsi(original_svs, sr_image_path, output_wsi, method="PIL")
sr_to_wsi(original_svs, sr_image_path, output_wsi_vips, method="pyvips")
