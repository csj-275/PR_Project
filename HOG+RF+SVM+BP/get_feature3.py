import cv2
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2

# 读入一组图像
# 读取图像
filename = './face/rawdata/1224'
with open(filename, 'rb') as f:
    content = f.read()
# 将字节数组转成 128*128 的图像
data = np.frombuffer(content, dtype=np.uint8)
img1 = data.reshape(128, 128)
filename = './face/rawdata/1225'
with open(filename, 'rb') as f:
    content = f.read()
# 将字节数组转成 128*128 的图像
data = np.frombuffer(content, dtype=np.uint8)
img2 = data.reshape(128, 128)
filename = './face/rawdata/1227'
with open(filename, 'rb') as f:
    content = f.read()
# 将字节数组转成 128*128 的图像
data = np.frombuffer(content, dtype=np.uint8)
img3 = data.reshape(128, 128)

# 初始化 SIFT 特征提取器
sift = cv2.xfeatures2d.SIFT_create()

# 提取 SIFT 特征
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
kp3, des3 = sift.detectAndCompute(img3, None)
print(len(kp1))
print(len(kp2))
print(len(kp3))
# 将所有特征堆叠到一个数组中
descriptors = np.vstack((des1, des2, des3))

# 计算每个特征的卡方统计量
k_best = SelectKBest(chi2, k=50)
k_best.fit_transform(descriptors, np.array([1, 2, 3]*len(kp1)))

# 选择最优特征
selected_features = k_best.get_support(indices=True)
