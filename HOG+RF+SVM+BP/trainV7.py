# BP Net

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
from sklearn.datasets import load_digits
import cv2
from skimage.feature import hog
import math
from my_fun import non_image
from get_all_label import sex_list, age_list, race_list, face_list
# 定义最大灰度级数
gray_level = 16


def maxGrayLevel(img):
    max_gray_level = 0
    (height, width) = img.shape
    print
    height, width
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    return max_gray_level + 1


def getGlcm(input, d_x, d_y):
    srcdata = input.copy()
    ret = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height, width) = input.shape

    max_gray_level = maxGrayLevel(input)

    # 若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i] * gray_level / max_gray_level

    for j in range(height - d_y):
        for i in range(width - d_x):
            rows = srcdata[j][i]
            cols = srcdata[j + d_y][i + d_x]
            ret[rows][cols] += 1.0

    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j] /= float(height * width)

    return ret


def feature_computer(p):
    Con = 0.0
    Eng = 0.0
    Asm = 0.0
    Idm = 0.0
    for i in range(gray_level):
        for j in range(gray_level):
            Con += (i - j) * (i - j) * p[i][j]
            Asm += p[i][j] * p[i][j]
            Idm += p[i][j] / (1 + (i - j) * (i - j))
            if p[i][j] > 0.0:
                Eng += p[i][j] * math.log(p[i][j])
    return Asm, Con, -Eng, Idm


def GLCM(img):
    glcm = getGlcm(img, 1, 0)
    asm, con, eng, idm = feature_computer(glcm)
    feature = np.array([asm, con, eng, idm])
    return feature

def extract_hog_features(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), visualize=False, multichannel=True, feature_vector=True, scales=[0.5, 1, 1.5]):
    hogs = []
    for scale in scales:
        # 缩放图片
        resized_img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
        # HOG特征提取
        hog_features = hog(resized_img, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=visualize)
        hogs.append(hog_features)
    # 将多个尺度的特征拼接在一起
    features = np.concatenate(hogs)
    return features.tolist()





def load_data():
    features = []  # 存取图像特征
    i = 1223
    size_image = [2412, 2416]
    # while i <= 5222:
    while i <= 5222:
        if i in non_image or i in size_image:
            pass
        else:
            filename = './face/rawdata/' + str(i)
            with open(filename, 'rb') as f:
                content = f.read()
            data = np.frombuffer(content, dtype=np.uint8)
            img = data.reshape(128, 128)
            feature = extract_hog_features(img)
            features.append(feature)
            print(f'第{i}张图片提取特征成功！')
        i += 1


    Y = sex_list
    X = np.array(features).reshape(len(features), len(features[0]))  # 修正输入

    return X, Y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size # 输入层
        self.hidden_size = hidden_size # 隐藏层
        self.output_size = output_size # 输出层
        self.W1 = np.random.randn(input_size, hidden_size) # 权重1
        self.b1 = np.zeros((1, hidden_size)) # 偏置1
        self.W2 = np.random.randn(hidden_size, output_size) # 权重2
        self.b2 = np.zeros((1, output_size)) # 偏置2

    def feedforward(self, X): # 反馈
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = softmax(self.z2)
        return self.a2

    def backpropagation(self, X, y, lr): # 反向传播
        m = X.shape[0]
        dZ2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dZ1 = np.dot(dZ2, self.W2.T) * sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train(self, X, y, lr, epochs): # 训练
        for i in range(epochs):
            y_pred = self.feedforward(X)
            loss = np.sum(-y * np.log(y_pred))
            self.backpropagation(X, y, lr)
            if i % 10 == 0:
                print("Epoch:", i, "Loss:", loss)

    def predict(self, X): # 预测
        y_pred = self.feedforward(X)
        return np.argmax(y_pred, axis=1)


# 加载数据
X, y = load_data()
# X = X / 10.0  # 归一化
print('照片加载完成')

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化神经网络
input_size = X.shape[1]
hidden_size = 64
output_size = len(np.unique(y))
nn = NeuralNetwork(input_size, hidden_size, output_size)

# 训练神经网络
lr = 0.3
epochs = 500
nn.train(X_train, np.eye(output_size)[y_train], lr, epochs)

# 评价网络分类情况
y_pred = nn.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

# Show some predictions
# for i in range(10):
#     plt.imshow(X_test[i], cmap="gray")
#     plt.title("Label: %d, Predicted: %d" % (y_test[i], y_pred[i]))
#     plt.show()