#
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import re
from sklearn.model_selection import train_test_split
import math
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
    # # 设置GLCM算法的参数
    # distances = [1]  # 矩阵距离
    # angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 矩阵角度
    # levels = 256  # 灰度级数
    # symmetric = True  # 对称矩阵
    # normed = True  # 归一化
    # # 计算灰度共生矩阵
    # glcm = cv2.calcGLCM(img, distances, angles, levels, symmetric, normed)
    # # 提取GLCM特征
    # contrast = cv2.getGcmFeature(glcm, cv2.GLCM_CONTRAST)
    # dissimilarity = cv2.getGcmFeature(glcm, cv2.GLCM_DISIMILARITY)
    # homogeneity = cv2.getGcmFeature(glcm, cv2.GLCM_HOMOGENEITY)
    # energy = cv2.getGcmFeature(glcm, cv2.GLCM_ENERGY)
    # correlation = cv2.getGcmFeature(glcm, cv2.GLCM_CORRELATION)



def extract_description(filename, all): # 提取特征描述
    non_image = [1228, 1232, 1808, 4056, 4135, 4136, 5004]  # 没有数据特征的图片编号
    non_image = [str(i) for i in non_image]
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            pattern = r'\((.*?)\s+(.*?)\)'  # 匹配形式
            items = re.findall(pattern, line)
            d = {k.strip('_'): v.strip("'") for k, v in items}  # 将特征描述存入字典
            nums = re.findall(r'\d+', line)  # 找出数字
            nums = ''.join(nums)
            if nums not in non_image:
                d['prop'] = d['prop'] + ')'
                all[nums] = d
                # print(f'{nums}: {all[nums]}')  # 打印字典
            line = f.readline()
    return all

def load_data():
    features = []  # 存取图像特征
    i = 1223
    non_image = [1228, 1232, 1808, 4056, 4135, 4136, 5004]  # 没有数据特征的图片编号
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
            feature = GLCM(img)
            features.append(feature.tolist())
            print(f'第{i}张图片提取特征成功！')
        i += 1

    all = dict()  # 存放所有数据特征
    all = extract_description('./face/faceDR', all)
    all = extract_description('./face/faceDS', all)
    age = []
    for key in all.keys():
        if int(key) in non_image or int(key) in size_image:
            pass
        else:
            if all[key]['age'] == 'senior':
                age.append(3)
            elif all[key]['age'] == 'adult':
                age.append(2)
            elif all[key]['age'] == 'teen':
                age.append(1)
            else:
                age.append(0)

    Y = np.array(age)
    X = np.array(features).reshape(len(features), len(features[0]))  # 修正输入

    return X, Y


# 加载数据
X,Y = load_data()
# 划分数据集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
print('划分数据集完成')

# svm 0.72 0.84 0.64
svm =SVC(C=2, kernel='poly', degree=10, gamma=20, decision_function_shape='ovr')      #acc=0.9534
svm.fit(X_train, Y_train)
Y_predict = svm.predict(X_test)













































