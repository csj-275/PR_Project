# 测试神经网络提取特征


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from get_all_label import sex_list, age_list, face_list, race_list
from my_fun import get_img


def load_data(): # 加载数据
    X = []  # 存取图像特征
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
            X.append(img)
        i += 1
    X = np.array(X)
    print('加载数据成功')
    return X
y = sex_list

# 使用神经网络进行特征提取
model = Sequential([
    Dense(128, activation='relu', input_dim=784),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, np.eye(10)[y], epochs=10, batch_size=32)

# 使用 t-SNE 进行降维
pca = PCA(n_components=50)
scaler = StandardScaler()
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
pipeline = make_pipeline(pca, scaler, tsne)
X_tsne = pipeline.fit_transform(X)

# 可视化降维后的结果
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
plt.title('t-SNE visualization of MNIST dataset')
plt.show()
