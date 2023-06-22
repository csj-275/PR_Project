# hog图像特征提取并可视化
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import cv2
import numpy as np
def get_img(i):
    filename = './face/rawdata/' + str(i)
    with open(filename, 'rb') as f:
        content = f.read()
    data = np.frombuffer(content, dtype=np.uint8)
    img = data.reshape(128, 128)
    return img
# 读取图像
image= get_img(3512)
k = 8
c = 8
# 提取HOG特征
# fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
#                     cells_per_block=(2, 2), visualize=True, multichannel=False)
fd, hog_image = hog(image, orientations=9, pixels_per_cell=(k, k), cells_per_block=(c, c),visualize=True)
            # block_norm='L2-Hys',
# 显示原始图像
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# 显示HOG特征图像
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')

plt.show()
