# coding=utf-8
import openslide
import numpy as np
import scipy.misc
import cv2
import pyvips
import numpy as np
from PIL import Image
import time
import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
Image.MAX_IMAGE_PIXELS = None
import multiprocessing
from tqdm import tqdm


class WholeSlideImage(object):
    def __init__(self, path):

        """
        Args:
            path (str): fullpath to WSI file
        """

        #         self.name = ".".join(path.split("/")[-1].split('.')[:-1])
        self.path = path
        self.name = os.path.splitext(os.path.basename(self.path))[0]
        self.wsi = openslide.open_slide(self.path)
        # img with max mag
        #self.img = pyvips.Image.new_from_file(self.path, level=0)
        self.level_downsamples = self._assertLevelDownsamples()
        self.level_dim = self.wsi.level_dimensions
        self.lr_level = self._getLRLevel()
        self.scale = 4


    def getOpenSlide(self):
        return self.wsi

    def get_lr_image(self):
        lr_img = self.wsi.read_region((0, 0), self.lr_level, self.level_dim[self.lr_level])
        return lr_img.convert("RGB")

    def get_hr_size(self):
        return self.level_dim[0][0], self.level_dim[0][1]

    def _assertLevelDownsamples(self):
        level_downsamples = []
        dim_0 = self.wsi.level_dimensions[0]

        for downsample, dim in zip(self.wsi.level_downsamples, self.wsi.level_dimensions):
            estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
            level_downsamples.append(estimated_downsample) if estimated_downsample != (
                downsample, downsample) else level_downsamples.append((downsample, downsample))

        return level_downsamples

    def _getLRLevel(self):
        level = 0
        for i in range(len(self.level_dim)):
            if abs(self.wsi.level_downsamples[level] - 4.0) < 1:
                return level
            else:
                level += 1
        return None

    def _checkMagIs40x(self):
        pass

    def getRegion(self, x, y, w, h, needlr=True):
        patch = self.wsi.read_region((x, y), 0, (w, h))
        if needlr:
            patch_sr = self.wsi.read_region((x, y), self.lr_level, (w // self.scale, h // self.scale))
            return np.array(patch), np.array(patch_sr)
        return np.array(patch)

    def get_hrlr_pair(self):
        return self.img, self.lr_img

    def get_mpp(self):
        """
        获取单个 WSI 的 MPP（microns per pixel）值。

        Args:
            wsi_path (str): WSI 文件路径
            custom_mpp_keys (list[str], optional): 用户自定义的 MPP 属性键列表

        Returns:
            float or None: 返回 MPP 值（microns/pixel），无法获取返回 None
        """
        try:
            slide = openslide.OpenSlide(self.path)
            props = slide.properties

            # 常用 MPP 属性键
            mpp_keys = [
                openslide.PROPERTY_NAME_MPP_X,  # 'openslide.mpp-x'
                'openslide.mirax.MPP',
                'aperio.MPP',
                'hamamatsu.XResolution',
                'openslide.comment',
                'openslide.mpp-x',
            ]

            # 尝试从属性中获取 MPP
            for key in mpp_keys:
                if key in props:
                    try:
                        mpp = float(props[key])
                        return mpp
                    except ValueError:
                        continue

            # 尝试从 TIFF 头部信息中获取
            x_resolution = props.get('tiff.XResolution')
            unit = props.get('tiff.ResolutionUnit')
            if x_resolution and unit:
                try:
                    if unit.lower() == 'centimeter':
                        mpp = 10000 / float(x_resolution)
                        return mpp
                    elif unit.lower() in ['inch', 'inches']:
                        mpp = 25400 / float(x_resolution)
                        return mpp
                except ValueError:
                    pass

            # 所有方法失败
            print(f"Warning: 无法从 {self.path} 中提取 MPP")
            return None

        except Exception as e:
            print(f"Error processing {self.path}: {str(e)}")
            return None

        finally:
            if 'slide' in locals():
                slide.close()


# wsi_path = "/media/dell/data/zhangv1/WSISR/WSI_Level_SR/wsisr_pipeline/preset/datasets/tcga_l/test/wsi/TCGA-2Y-A9GU-01Z-00-DX1.700CBBD7-9F58-470D-A85B-C7BAE5608018.svs"
# wsi = WholeSlideImage(wsi_path)
# slide = wsi.getOpenSlide()
# # hr, lr = wsi.getRegion(25000, 25000, 512, 512, needlr=True)
# # cv2.imwrite(f'../hr.png', cv2.cvtColor(hr, cv2.COLOR_RGB2BGR))
# # cv2.imwrite(f'../lr.png', cv2.cvtColor(lr, cv2.COLOR_RGB2BGR))
# print(wsi.level_dim[1])
# print(wsi.level_downsamples[1])
# print(wsi.lr_level)
