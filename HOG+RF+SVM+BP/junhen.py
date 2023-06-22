
import cv2
import numpy as np
import matplotlib.pyplot as plt
from my_fun import get_img
# 加载图片
img = get_img(5129)
# 进行直方图均衡化
equ = cv2.equalizeHist(img)

# 显示原图和均衡化后的图像
plt.figure(figsize=(8,8))
plt.subplot(2, 1, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 1, 2)
plt.imshow(equ, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

# 显示图像直方图
plt.figure(figsize=(8,4))
plt.hist(img.ravel(),256,[0,256])
plt.title('Original Histogram')
plt.show()

plt.figure(figsize=(8,4))
plt.hist(equ.ravel(),256,[0,256])
plt.title('Equalized Histogram')
plt.show()
