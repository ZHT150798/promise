import os
import openslide

# 设置路径
folder = '/media/dell/One Touch/shandong/'

# 支持的扩展名
slide_exts = {'.svs', '.tiff', '.ndpi', '.mrxs'}

# 遍历文件夹
for fname in os.listdir(folder):
    fpath = os.path.join(folder, fname)
    ext = os.path.splitext(fname)[1].lower()

    if os.path.isfile(fpath) and ext in slide_exts:
        try:
            slide = openslide.OpenSlide(fpath)
            downsamples = slide.level_downsamples  # list of floats
            print(f"\nSlide: {fname}")
            for i, d in enumerate(downsamples):
                print(f"  Level {i}: downsample factor = {d}")
        except Exception as e:
            print(f"Failed to read {fname}: {e}")
