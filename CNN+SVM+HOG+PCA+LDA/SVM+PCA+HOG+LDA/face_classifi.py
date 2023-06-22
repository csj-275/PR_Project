import shutil
import cv2
import os

# 读入分类的标签txt文件
# label_file = open("D:\\2023PR\\face\\faceDR_txt.txt", 'r')
# 第一个标签
label_file1 = 'faceDR_txt.txt'
# 第二个标签
label_file2 = 'faceDS_txt.txt'
# 原始文件的根目录
# input_path = "E:\\pythonProject\\data\\cars_train"
input_path = 'D:/2023PR/face/new_pgmdata'
# 保存文件的根目录
# output_path = "E:\\pythonProject\\result"
destination_path1 = 'D:/2023PR/face/sex/male'
destination_path2 = 'D:/2023PR/face/sex/female'
# destination_path1 = 'D:/2023PR/face/testdata/male'
# destination_path2 = 'D:/2023PR/face/testdata/female'

def mycopyfile(srcfile, dstpath,flag=False):  # 复制函数
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        shutil.copy(srcfile, dstpath)  # 复制文件
        print("copy %s -> %s" % (srcfile, dstpath))
        flag =True
    return flag

female = 0
male = 0
label_files = [label_file1,label_file2]
# 一行行读入标签文件
for file in label_files:
    label_file = open(file,'r')
    data = label_file.readlines()
# 按男女分
    for line in data:
        str1 = line.split(' ')
        # print(str1)
        if 'male)' in str1:
            file = input_path + '/' + str1[1]+'.pgm'
            flag = mycopyfile(file,destination_path1)
            # 统计男性人数
            if flag:
                male += 1
        elif 'female)' in str1:
            file = input_path + '/' + str1[1]+'.pgm'
            flag = mycopyfile(file, destination_path2)
            # 统计女性人数
            if flag:
                female += 1

# 完成提示
print('男性%d人'% male)
print('女性%d人'% female)
print('总共%d人'% (male+female))
print('完成')



