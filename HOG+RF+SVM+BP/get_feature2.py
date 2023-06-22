import cv2
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# 读取图像
filename = './face/rawdata/1224'
with open(filename, 'rb') as f:
    content = f.read()
# 将字节数组转成 128*128 的图像
data = np.frombuffer(content, dtype=np.uint8)
img = data.reshape(128, 128)

# 将图像展平为一维向量
img_vec = img.flatten()

# 使用PCA算法进行特征提取
pca = PCA(n_components=2)
img_features = pca.fit_transform(img_vec.reshape(1, -1))

# 可视化特征
plt.scatter(img_features[:, 0], img_features[:, 1])
plt.show()
