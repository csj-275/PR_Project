# 可视化lda特征
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import re
import sklearn
from my_fun import non_image
from get_all_label import sex_list, age_list, face_list, race_list


def LDA(X, y):
    X1 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
    X2 = np.array([X[i] for i in range(len(X)) if y[i] == 1])

    len1 = len(X1)
    len2 = len(X2)

    mju1 = np.mean(X1, axis=0)  # 求中心点
    mju2 = np.mean(X2, axis=0)

    cov1 = np.dot((X1 - mju1).T, (X1 - mju1))
    cov2 = np.dot((X2 - mju2).T, (X2 - mju2))
    Sw = cov1 + cov2

    w = np.dot(np.mat(Sw).I, (mju1 - mju2).reshape((len(mju1), 1)))  # 计算w
    X1_new = func(X1, w)
    X2_new = func(X2, w)
    y1_new = [1 for i in range(len1)]
    y2_new = [2 for i in range(len2)]
    return X1_new, X2_new, y1_new, y2_new


def extract_description(filename, all): # 提取特征描述
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
    feature_all = []  # 存取图像特征
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
            feature_all.append(img.flatten().tolist())
            print(f'第{i}张图片提取特征成功！')

        i += 1

    X = np.array(feature_all).reshape(len(feature_all), len(feature_all[0]))
    Y = face_list
    # 创建LDA对象并拟合数据
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X, Y)
    return X_lda, Y

X_lda, y = load_data()

# 可视化LDA特征
colors = ['black', 'blue', 'purple', 'yellow']
#colors = [color[i] for i in range(max(y)+1)]
#
plt.figure(figsize=(10, 8))
for i in range(len(colors)):
    plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], c=colors[i], label=str(i),
                alpha=0.7, edgecolors='none')

size_image = [2412, 2416]
txt = [i for i in range(1223, 5223) if i not in non_image and i not in size_image]
for i in range(4000-len(non_image)-2):
    plt.annotate(txt[i], (X_lda[i,0], X_lda[i,1]))

plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of Handwritten Digits')
plt.show()
