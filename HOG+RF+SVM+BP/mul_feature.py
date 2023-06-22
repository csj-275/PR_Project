# 融合特征提取
# import cv2
# import numpy as np
# from skimage.feature import local_binary_pattern
# from sklearn.decomposition import PCA
# from my_fun import get_img
# img = get_img(1223)
# # 提取lbp特征
# lbp = local_binary_pattern(img, 8, 1)
# lbp_hist, _ = np.histogram(lbp, bins=256)
# # 提取HOG特征
# hog = cv2.HOGDescriptor()
# hog_feat = hog.compute(img)
# hog_feat = hog_feat.ravel()
# # 将多种特征融合起来
# pca = PCA(n_components=50)
# print(lbp_hist.size)
# print(len(hog_feat))
# X = np.hstack((hog_feat, lbp_hist.ravel()))
# print(X.size)
# X_pca = pca.fit_transform(X.reshape(1, -1))
# print(X_pca)
#
#
#
# import cv2
# import numpy as np
# from skimage.feature import hog
# from skimage import exposure
# from skimage.feature import local_binary_pattern
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
#
# # 加载图像数据
# img1 = cv2.imread('image1.jpg', 0)
# img2 = cv2.imread('image2.jpg', 0)
# img3 = cv2.imread('image3.jpg', 0)
#
# # 提取HOG特征
# fd, hog_image1 = hog(img1, orientations=8, pixels_per_cell=(16, 16),
#                       cells_per_block=(1, 1), visualize=True, multichannel=False)
#
# # 提取LBP特征
# lbp_radius = 1
# lbp_n_points = 8 * lbp_radius
# lbp1 = local_binary_pattern(img1, lbp_n_points, lbp_radius, method='uniform')
# lbp2 = local_binary_pattern(img2, lbp_n_points, lbp_radius, method='uniform')
# lbp3 = local_binary_pattern(img3, lbp_n_points, lbp_radius, method='uniform')
#
# # 将HOG和LBP特征融合在一起
# feature1 = np.concatenate((fd1, np.histogram(lbp1.ravel(), bins=256)[0]))
# feature2 = np.concatenate((fd2, np.histogram(lbp2.ravel(), bins=256)[0]))
# feature3 = np.concatenate((fd3, np.histogram(lbp3.ravel(), bins=256)[0]))
#
# # 将所有特征向量组合成一个二维矩阵
# X = np.vstack((feature1, feature2, feature3))
#
# # 标准化数据
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
#
# # 使用PCA进行特征降维
# pca = PCA(n_components=100)
# X_pca = pca.fit_transform(X)
#
# # 查看特征向量的维度
# print(X_pca.shape)
import numpy as np
from my_fun import hog_before, get_img, print_result, hog_before, hog_improve, non_image
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


from get_all_label import sex_list, age_list, race_list, face_list # 加载标签
from sklearn.model_selection import train_test_split # 划分数据集
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix # 计算得分
from sklearn.ensemble import RandomForestClassifier # 随机森林
from sklearn.tree import DecisionTreeClassifier # 决策树
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.linear_model import LogisticRegression # 逻辑回归
from sklearn.naive_bayes import GaussianNB # 朴素贝叶斯
from sklearn.svm import SVC # svm

def train(X, Y, method):
    # 划分数据集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    # print('划分数据集完成')
    if method == 'knn':
        knn = KNeighborsClassifier(n_neighbors=2)  # 0.93
        knn.fit(X_train, Y_train)
        print(knn)
        print('测试数据集得分：{:.2f}'.format(knn.score(X_test, Y_test)))
        Y_predict = knn.predict(X_test)
    elif method == 'svm':
        svm = SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovr')      #acc=0.9534
        svm.fit(X_train, Y_train)
        Y_predict = svm.predict(X_test)
    elif method == 'DecisionTree':
        tree_D = DecisionTreeClassifier()
        tree_D.fit(X_train, Y_train)
        Y_predict = tree_D.predict(X_test)
    elif method == 'Logistic':
        logistic = LogisticRegression()
        logistic.fit(X_train, Y_train)
        Y_predict = logistic.predict(X_test)
    elif method == 'GaussianNB':
        mlt=GaussianNB()
        mlt.fit(X_train, Y_train)
        Y_predict = mlt.predict(X_test)
    elif method == 'RandomForest':
        Forest = RandomForestClassifier(max_depth=30, max_features='sqrt', min_samples_leaf=1, min_samples_split=2, n_estimators=200)
        Forest.fit(X_train, Y_train)
        Y_predict = Forest.predict(X_test)
    return X_test, Y_test, Y_predict


def load_data(): # 加载数据
    feature_all = []  # 存取图像特征
    i = 1223
    size_image = [2412, 2416]
    # while i <= 5222:
    lbp_radius = 1
    lbp_n_points = 8 * lbp_radius
    while i <= 5222:
        if i in non_image or i in size_image:
            pass
        else:
            filename = './face/rawdata/' + str(i)
            with open(filename, 'rb') as f:
                content = f.read()
            data = np.frombuffer(content, dtype=np.uint8)
            img = data.reshape(128, 128)
            f1 = hog_before(img)
            f2 = local_binary_pattern(img, lbp_n_points, lbp_radius, method='uniform')
            feature = np.concatenate((f1, np.histogram(f2.ravel(), bins=256)[0]))
            feature_all.append(feature)
            print(f'第{i}张图片提取特征成功！')
        i += 1
    X = np.array(feature_all)
    print('加载数据成功')
    return X

X = load_data()
print(X.shape)
# 标准化数据
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 使用PCA进行特征降维
pca = PCA(n_components=200)
X_pca = pca.fit_transform(X)


method = ['knn', 'svm', 'DecisionTree', 'Logistic', 'GaussianNB', 'RandomForest']
Y_list = [sex_list, age_list, race_list, face_list]
Y_name = ['sex', 'age', 'race', 'face']
acc_list = []
pre_list = []
rec_list = []
for i in method:
    acc_temp = []
    pre_temp = []
    rec_temp = []
    for j in range(4):
        X_test, Y_test, Y_predict = train(X_pca, Y_list[j], i)
        acc, precision, recall = print_result(Y_test, Y_predict, 'method:'+i+' -- class:'+Y_name[j])
        acc_temp.append(acc)
        pre_temp.append(precision)
        rec_temp.append(recall)
    acc_list.append(acc_temp)
    pre_list.append(pre_temp)
    rec_list.append(rec_temp)
print('==============准确率==============')
for i in acc_list:
    print(', '.join(i))
print('==============精确率==============')
for i in pre_list:
    print(', '.join(i))
print('==============召回率==============')
for i in rec_list:
    print(', '.join(i))
