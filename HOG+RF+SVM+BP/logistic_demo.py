# 逻辑回归参数优化
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from get_all_label import sex_list, age_list, race_list, face_list # 加载标签
import numpy as np
from my_fun import hog_before, hog_improve, non_image, print_result
from sklearn.ensemble import RandomForestClassifier # 随机森林
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.linear_model import LogisticRegression # 逻辑回归

def load_data():
    features_1 = []  # 存取图像特征
    features_2 = []
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
            features_1.append(hog_before(img))
            # features_2.append(hog_improve(img))
            print(f'第{i}张图片提取特征成功！')
        i += 1
    X1 = np.array(features_1).reshape(len(features_1), len(features_1[0]))  # 修正输入
    # X2 = np.array(features_2).reshape(len(features_2), len(features_2[0]))  # 修正输入
    print('加载数据成功')
    return X1

X1 = load_data()
X_train, X_test, Y_train, Y_test = train_test_split(X1, sex_list, test_size=0.2, random_state=42)
param_grid = {'C': [0.1, 1, 5, 10],
              'class_weight': [None, 'balanced'],
              'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag'],
              'tol': [0.06, 0.1, 0.14, 0.18]}
acc_arr = []
rec_arr = []
pre_arr = []

for i in range(1):
    a = []
    p = []
    r = []
    for j in range(1):
        # C = param_grid['C'][i]
        # tol = param_grid['tol'][j]
        logistic = LogisticRegression()
        logistic.fit(X_train, Y_train)
        Y_predict = logistic.predict(X_test)
        acc, precision, recall = print_result(Y_test, Y_predict, ' ======= ')
        a.append(str(acc))
        p.append(str(precision))
        r.append(str(recall))
    acc_arr.append(a)
    pre_arr.append(p)
    rec_arr.append(r)

print('==============准确率==============')
for i in acc_arr:
    print(','.join(i))
print('==============精确率==============')
for i in pre_arr:
    print(', '.join(i))
print('==============召回率==============')
for i in rec_arr:
    print(', '.join(i))
# model = RandomForestClassifier()
# param_grid = { 'n_estimators': [10, 50, 100, 200],
#               'max_depth': [None, 10, 20, 30],
#               'min_samples_split': [2, 5, 10],
#               'min_samples_leaf': [1, 2, 4],
#               'max_features': ['sqrt', 'log2']
# }
# param_grid = {'n_neighbors': [2,3,4,5]}
# model = KNeighborsClassifier()  # 0.93
# grid_model = GridSearchCV(model, param_grid=param_grid)
# grid_model.fit(X_train, y_train)
# print("最佳参数: {}".format(grid_model.best_params_))
# print("最佳得分: {:.2f}".format(grid_model.best_score_))
# print('最佳模型寻找完成')
# best_model = KNeighborsClassifier(n_neighbors=grid_model.best_params_['n_neighbors'])
# best_model.fit(X_train, y_train)
# print("测试分数: {:.2f}".format(best_model.score(X_test, y_test)))
