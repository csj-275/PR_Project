# 批量提取特征
import numpy as np
import cv2
from skimage.feature import hog

def extract_hog_features_single(X):
    image_descriptors_single = []
    fd, _ = hog(X, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(16, 16),
                            block_norm='L2-Hys', visualize=True)
    image_descriptors_single.append(fd)
    return image_descriptors_single


feature_all = [] # 存取图像特征
i = 1223
non_image = [1228, 1232, 1808, 4056, 4135, 4136, 5004]  # 没有数据特征的图片编号
size_image = [2412, 2416]
#while i <= 5222:
while i <= 5222:
    if i not in non_image or i not in size_image:
        filename = './face/rawdata/' + str(i)
        with open(filename, 'rb') as f:
            content = f.read()
        data = np.frombuffer(content, dtype=np.uint8)
        img = data.reshape(128, 128)
        feature = extract_hog_features_single(img)
        feature_all.append(feature)
        print(f'第{i}张图片提取特征成功！')

    i += 1
X = np.array(feature_all).reshape(len(feature_all), len(feature_all[0][0])) # 修正输入
print(len(feature_all[0][0]))
print(X.shape)

