import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, LatentDirichletAllocation

# 加载手写数字数据集，其中包含 1797 张 8x8 的灰度图像
digits = load_digits()

# 使用 StandardScaler 对图像数据进行标准化
X = StandardScaler().fit_transform(digits.data)

# 使用 PCA 将 64 维的图像特征降到 20 维
pca = PCA(n_components=20)
X_pca = pca.fit_transform(X)

# 使用 LDA 将 20 维的特征降到 2 维，并可视化降维后的图像
lda = LatentDirichletAllocation(n_components=2)
lda.fit(X_pca)
X_lda = lda.transform(X_pca)

plt.figure(figsize=(10, 6))
for i in range(10):
    plt.scatter(X_lda[digits.target == i, 0],
                X_lda[digits.target == i, 1],
                label=str(i))
plt.legend()
plt.xlabel('Topic 1')
plt.ylabel('Topic 2')
plt.title('LDA Visualization of Digits')
plt.show()
