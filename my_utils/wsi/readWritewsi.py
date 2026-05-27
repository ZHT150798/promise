# coding=utf-8
import openslide
import numpy as np
import scipy.misc
# import cv2
import numpy as np
from PIL import Image
import time
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
Image.MAX_IMAGE_PIXELS = None
import cv2
import pyvips


# slide = openslide.open_slide('../preset/dataset/Bladder (10).svs')
# level = 0
# print(slide.level_count)
# print(slide.level_dimensions)
# W, H = slide.level_dimensions[level]
# print(slide.properties)
# start = time.time()
# img = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]))
level = 1
start = time.time()
slide = openslide.open_slide('../preset/Bladder (10).svs')
img2 = pyvips.Image.new_from_file('../preset/Bladder (10).svs', level=level)
image1 = img2.crop(0, 0, 8192, 8192)
image = slide.read_region((0, 0), level, (8192, 8192))# slide.level_dimensions[level]
end1 = time.time()
img2 = np.array(image)
img1 = np.array(image1)  # slide.dimensions
print(np.mean((img2 - img1) ** 2))
cv2.imwrite(f'../openslide.png', cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
cv2.imwrite(f'../pyvips.png', cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
end = time.time()
output_path = r'../../wsi_png'
if not os.path.exists(output_path):
    os.mkdir(output_path)
print('read time', (end1 - start))
print('save time', (end - end1))
# #PIL.Image.fromarray(img).save('../wsi_png/Bladder (10).png')
# cv2.imwrite(f'../wsi_png/Bladder (10)_cv2_{level}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
# # slide.close()
#
# end = time.time()
# print('read time', (end1 - start))
# print('save time', (end - end1))
# start = time.time()
# #img = cv2.imread("../wsi_png/Bladder (10)_cv2_1.png")
# img = Image.open("../wsi_png/Bladder (10)_cv2_1.png").convert('RGB')
# img_array = np.array(img)
# end = time.time()
# print('read time', (end - start))