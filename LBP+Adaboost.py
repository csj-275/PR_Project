import re
import cv2
import numpy as np
import glob
import sklearn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
import time
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib

#读取并生成标签文件
labels=[i.strip() for i in open("faceDR").readlines()]+[i.strip() for i in open("faceDS").readlines()]

# 提取标签信息，定义文本
text = "1223 (_sex  male) (_age  child) (_race white) (_face smiling) (_prop '())"
def getres(text):
    # 定义正则表达式模式
    pattern = r"_sex  (\w+).*_age  (\w+).*_race (\w+).*_face (\w+).*_prop"

    # 使用正则表达式匹配文本
    match = re.search(pattern, text)

    if match:
        # 提取性别、年龄、种族等信息
        gender = match.group(1)
        age = match.group(2)
        race = match.group(3)
        face = match.group(4)

        # 打印提取的信息
#         print("性别:", gender)
#         print("年龄:", age)
#         print("种族:", race)
#         print("表情:", face)
        return gender,age,race,face
    else:
        return []
d={}
for i in labels:
     tmp=getres(i)
     if tmp and len(tmp)==4:
        d[i.split(" ")[0]]=tmp

#LBP特征提取
tar="gender"
def get_lbp_feature(image):
    image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    lbp_feature = []
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            lbp_code = ''
            center = int(gray[i, j])
            for k in range(8):
                if gray[i - 1 + (k // 3), j - 1 + (k % 3)] >= center:
                    lbp_code += '1'
                else:
                    lbp_code += '0'
            lbp_feature.append(int(lbp_code, 2))
    hist, _ = np.histogram(lbp_feature, bins=256)
    return hist
lbpfea=get_lbp_feature("pgmdata/1223.pgm")

#构建标签数据集

fea=[]
labels=[]
index= 0
for i in tqdm(glob.glob("pgmdata\\*")):
    fea.append(get_lbp_feature(i))
    tmp=str(i.split("\\")[-1][:-4])
    labels.append(d[tmp][index])
#     break

#划分数据集
data=np.array(fea)
# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=32)
# 输出划分后的数据集大小
print("训练集大小:", X_train.shape)
print("测试集大小:", X_test.shape)

#Adaboost分类
st = time.time()
ada =OneVsRestClassifier(AdaBoostClassifier(n_estimators=2000))
ada.fit(X_train,y_train)

pred=ada.predict(X_test)
et=time.time()
print ("cost:",et-st,"s")

#分类结果
print (classification_report(y_test,pred,digits=4))

#得出混淆矩阵
labelnew=set(labels)

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

C2=sklearn.metrics.confusion_matrix(y_test,pred, sample_weight=None,labels=list(labelnew))
C2
#混淆矩阵可视化
sns.set()
f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(C2,annot=True,ax=ax,cmap="OrRd", fmt='.20g',xticklabels=list(labelnew),yticklabels=list(labelnew)) #画热力图
ax.set_title('confusion matrix') #标题

plt.savefig('cm.png',dpi=600)
plt.show()