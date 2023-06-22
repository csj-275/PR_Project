
# hog特征提取+训练(年龄)
import re
import numpy as np
import cv2
from skimage.feature import hog
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import os
import pickle
import seaborn
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from sklearn.tree import DecisionTreeClassifier # 决策树
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.linear_model import LogisticRegression # 逻辑回归
from sklearn.naive_bayes import GaussianNB # 朴素贝叶斯

def get_img(i):
    filename = './face/rawdata/' + str(i)
    with open(filename, 'rb') as f:
        content = f.read()
    data = np.frombuffer(content, dtype=np.uint8)
    img = data.reshape(128, 128)
    return img

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

# def extract_hog_features_single(X):
#     image_descriptors_single = []
#     fd, _ = hog(X, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(16, 16),
#                             block_norm='L2-Hys', visualize=True)
#     image_descriptors_single.append(fd)
#     return image_descriptors_single


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

def main():
    feature_all = [] # 存取图像特征
    i = 1223
    non_image = [1228, 1232, 1808, 4056, 4135, 4136, 5004]  # 没有数据特征的图片编号
    size_image = [2412, 2416]
    #while i <= 5222:
    while i <= 5222:
        if i in non_image or i  in size_image:
            pass
        else:
            img = get_img(i)
            feature = extract_hog_features(img)
            feature_all.append(feature)
            print(f'第{i}张图片提取特征成功！')

        i += 1
    X = np.array(feature_all).reshape(len(feature_all), len(feature_all[0])) # 修正输入
    print('特征提取完成')

    all = dict() # 存放所有数据特征
    all = extract_description('./face/faceDR', all)
    all = extract_description('./face/faceDS', all)
    age = []

    # Acc: 0.8216432865731463
    # Precision: 0.7142084207371892
    # Recall: 0.6090529600791001
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
    print('标签整理完成')

    # 划分数据集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    print('划分数据集完成')

    # knn 0.83 0.82 0.82
    knn = KNeighborsClassifier(n_neighbors=2)  # 0.93
    knn.fit(X_train, Y_train)
    print(knn)
    print('测试数据集得分：{:.2f}'.format(knn.score(X_test, Y_test)))
    Y_predict = knn.predict(X_test)

    # svm 0.72 0.84 0.64
    # svm =sklearn.svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovr')      #acc=0.9534
    # svm.fit(X_train, Y_train)
    # Y_predict = svm.predict(X_test)

    #决策树算法   0.7 0.68 0.68
    # tree_D = DecisionTreeClassifier()
    # tree_D.fit(X_train, Y_train)
    # Y_predict = tree_D.predict(X_test)

    #朴素贝叶斯分类   0.65 0.64 0.58
    # mlt=GaussianNB()
    # mlt.fit(X_train, Y_train)
    # Y_predict = mlt.predict(X_test)

    #逻辑回归分类 0.87 0.85 0.85
    # logistic = LogisticRegression()
    # logistic.fit(X_train, Y_train)
    # Y_predict = logistic.predict(X_test)

    # 随机森林 0.83 0.84 0.80
    # Forest = RandomForestClassifier(n_estimators=180, random_state=0)
    # Forest.fit(X_train, Y_train)
    # Y_predict = Forest.predict(X_test)

    acc = accuracy_score(Y_test, Y_predict)
    precision = precision_score(Y_test, Y_predict, average='macro')
    recall = recall_score(Y_test, Y_predict, average='macro')
    cm = confusion_matrix(Y_test, Y_predict)
    print(cm)
    print('Acc: ', acc)
    print('Precision: ', precision)
    print('Recall: ', recall)

    # xtick = ['male', 'female']
    xtick = ['senior', 'adult', 'teen', 'child']
    ytick = xtick

    f, ax = plt.subplots(figsize=(7, 5))
    ax.tick_params(axis='y', labelsize=15)
    ax.tick_params(axis='x', labelsize=15)

    seaborn.set(font_scale=1.2)
    plt.rc('font', family='Times New Roman', size=15)

    seaborn.heatmap(cm, fmt='g', cmap='Blues', annot=True, cbar=True, xticklabels=xtick, yticklabels=ytick, ax=ax)

    plt.title('Confusion Matrix', fontsize='x-large')

    plt.show()

    for i in range(len(Y_predict)):
        if Y_predict[i] != Y_test[i]:
            img = get_img(i+1223)
            text = 'real:' +  Y_test[i] +'  predict:'+ Y_predict[i]
            font = cv2.FONT_HERSHEY_SIMPLEX
            position = (20, 20)
            font_scale = 1
            thickness = 1
            color = (255, 255, 255)
            cv2.putText(img, text, position, font, font_scale, color, thickness)
            cv2.imshow(img)
            cv2.waitKey(0)

if __name__ == '__main__':
    main()
