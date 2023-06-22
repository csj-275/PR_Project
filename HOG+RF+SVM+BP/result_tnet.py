# t-SNE特征提取+训练所有模型结果
from get_all_label import sex_list, age_list, race_list, face_list # 加载标签
from sklearn.model_selection import train_test_split # 划分数据集
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix # 计算得分
from sklearn.ensemble import RandomForestClassifier # 随机森林
from sklearn.tree import DecisionTreeClassifier # 决策树
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.linear_model import LogisticRegression # 逻辑回归
from sklearn.naive_bayes import GaussianNB # 朴素贝叶斯
from sklearn.svm import SVC # svm
from my_fun import print_result, hog_before, hog_improve, get_img, non_image
import numpy as np
from skimage.feature import local_binary_pattern # LBP
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE
from sklearn.utils import check_random_state
# 重新定义 TSNE 类
class TSNEAlt(TSNE):
    def transform(self, X):
        self._check_initialized()
        X = self._check_array(X)
        if self.perplexity is None:
            self.perplexity = 30
        random_state = check_random_state(self.random_state)
        X_embedded = self._tsne(
            X, perplexity=self.perplexity, early_exaggeration=self.early_exaggeration,
            learning_rate=self.learning_rate, n_iter=self.n_iter,
            n_iter_without_progress=self.n_iter_without_progress,
            min_grad_norm=self.min_grad_norm,
            metric=self.metric, init=self.init, verbose=self.verbose,
            random_state=random_state, method=self.method,
            angle=self.angle
        )
        return X_embedded

def load_data():
    X1 = []
    i = 1223
    size_image = [2412, 2416]
    lbp_radius = 1
    lbp_n_points = 8 * lbp_radius
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
            X1.append(img.flatten())
            print(f'第{i}张图片提取特征成功！')
        i += 1

    X = StandardScaler().fit_transform(X1)

    return X

def train(X, Y, method):
    # 划分数据集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    tsne = TSNEAlt(n_components=2, perplexity=30, learning_rate=200)
    print(dir(tsne))
    X_train = tsne.fit_transform(X_train)
    X_test = tsne.transform(X_test)
    # print('划分数据集完成')
    if method == 'knn':
        knn = KNeighborsClassifier(n_neighbors=2)  # 0.93
        knn.fit(X_train, Y_train)
        print(knn)
        print('测试数据集得分：{:.2f}'.format(knn.score(X_test, Y_test)))
        Y_predict = knn.predict(X_test)
    elif method == 'svm':
        svm = SVC(C=10, kernel='rbf', gamma=1, decision_function_shape='ovr')      #acc=0.9534
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
        Forest = RandomForestClassifier(n_estimators=180, random_state=0)
        Forest.fit(X_train, Y_train)
        Y_predict = Forest.predict(X_test)
    return X_test, Y_test, Y_predict

X1 = load_data()
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
        X_test, Y_test, Y_predict = train(X1, Y_list[j], i)
        acc, precision, recall = print_result(Y_test, Y_predict, 'method:'+i+' -- class:'+Y_name[j])
        acc_temp.append(str(acc))
        pre_temp.append(str(precision))
        rec_temp.append(str(recall))
    acc_list.append(acc_temp)
    pre_list.append(pre_temp)
    rec_list.append(rec_temp)
print('==============准确率==============')
for i in acc_list:
    print(','.join(i))
print('==============精确率==============')
for i in pre_list:
    print(', '.join(i))
print('==============召回率==============')
for i in rec_list:
    print(', '.join(i))
