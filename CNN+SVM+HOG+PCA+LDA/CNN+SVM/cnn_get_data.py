#coding=utf-8
 
import os
import numpy as np
import cv2


def get_img_list(dirname,flag=0):
    # rootdir= os.path.abspath('./data/'+dirname+'/')
    rootdir = os.path.abspath(dirname)
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    files=[]
    for i in range(0,len(list)):
        path = os.path.join(rootdir,list[i])
        if os.path.isfile(path):
            files.append(path)
    return files

images=[]
labels=[]

def read_img(list,flag=0):
    for i in range(len(list)-1):
         if os.path.isfile(list[i]):
             images.append(cv2.imread(list[i]).flatten())
             labels.append(flag)

# read_img(get_img_list('./sex/male/'),[0,1])
# read_img(get_img_list('./sex/male/'), [0,1])
# read_img(get_img_list('./sex/female/'), [1,0])
# read_img(get_img_list('./age/child/'), [1,0,0,0])
# read_img(get_img_list('./age/teen/'), [0,1,0,0])
# read_img(get_img_list('./age/adult/'),[0,0,1,0])
# read_img(get_img_list('./age/senior/'),[0,0,0,1])
# read_img(get_img_list('./emotion/smiling/'), [1,0,0])
# read_img(get_img_list('./emotion/serious/'), [0,1,0])
# read_img(get_img_list('./emotion/funny/'),[0,0,1])
read_img(get_img_list('./race/white/'), [1,0,0,0,0])
read_img(get_img_list('./race/black/'), [0,1,0,0,0])
read_img(get_img_list('./race/asian/'),[0,0,1,0,0])
read_img(get_img_list('./race/hispanic/'),[0,0,0,1,0])
read_img(get_img_list('./race/other/'),[0,0,0,0,1])
# read_img(get_img_list('./data/female/'),[1,0])
# read_img(get_img_list('./data/male'),[0,1])

images = np.array(images)
labels = np.array(labels)
print(labels)

#重新打乱
permutation = np.random.permutation(labels.shape[0])
all_images = images[permutation,:]
all_labels = labels[permutation,:]
# all_labels = labels[permutation]
print(all_labels)

#训练集与测试集比例 8：2
train_total = all_images.shape[0]
train_nums= int(all_images.shape[0]*0.8)
test_nums = all_images.shape[0]-train_nums

images = all_images[0:train_nums,:]
print(images)
labels = all_labels[0:train_nums,:]
# labels = all_labels[0:train_nums]
print(labels)


test_images = all_images[train_nums:train_total,:]
test_labels = all_labels[train_nums:train_total,:]
# test_labels = all_labels[train_nums:train_total]

train_labels_svm = []
test_labels_svm = []
train_labels_array = np.argmax(labels,1)
train_labels_svm = train_labels_array.tolist()
test_labels_array = np.argmax(test_labels,1)
test_labels_svm = test_labels_array.tolist()

