# 图像特征提取
import numpy as np
import cv2
from skimage.feature import hog


def extract_hog_features_single(X):
    image_descriptors_single = []
    fd, _ = hog(X, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(16, 16),
                            block_norm='L2-Hys', visualize=True)
    image_descriptors_single.append(fd)
    return image_descriptors_single


# 二进制读取文件
filename = './face/rawdata/1224'
with open(filename, 'rb') as f:
    content = f.read()

# 将字节数组转成 128*128 的图像
data = np.frombuffer(content, dtype=np.uint8)
img = data.reshape(128, 128)
features = extract_hog_features_single(img)
print(features)