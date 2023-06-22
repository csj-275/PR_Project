import shutil
import cv2
import os

# 读入分类的标签txt文件
# label_file = open("D:\\2023PR\\face\\faceDR_txt.txt", 'r')
# 第一个标签
# label_file1 = open('faceDR_txt.txt', 'r')
label_file1 = 'faceDR_txt.txt'
# 第二个标签
# label_file2 = open('faceDS_txt.txt', 'r')
label_file2 = 'faceDS_txt.txt'
# 原始文件的根目录
# input_path = "E:\\pythonProject\\data\\cars_train"
# input_path = 'D:/2023PR/face/new_pgmdata'
input_path = './new_pgmdata/'
# 保存文件的根目录
# output_path = "E:\\pythonProject\\result"
destination_path1 = './emotion/smiling/'
destination_path2 = './emotion/serious/'
destination_path3 = './emotion/funny/'

def mycopyfile(srcfile, dstpath, flag = False):  # 复制函数
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        shutil.copy(srcfile, dstpath)  # 复制文件
        # 各人数
        flag = True
        print("copy %s -> %s" % (srcfile, dstpath))
        return flag

smiling = 0
serious = 0
funny = 0
# 一行行读入标签文件
label_files = [label_file1,label_file2]
for file in label_files:
    label_file = open(file,'r')
    data = label_file.readlines()
    for line in data:
        str1 = line.split(' ')
        # print(str1[1])
        if 'smiling)' in str1:
            file = input_path + str1[1] + '.pgm'
            flag = mycopyfile(file,destination_path1)
            # num += 1
            # 统计微笑人数
            if flag:
                smiling += 1
        elif 'serious)' in str1:
            file = input_path + str1[1] + '.pgm'
            flag = mycopyfile(file, destination_path2)
            # num += 1
            # 统计严肃人数
            if flag:
                serious += 1
        elif 'funny)' in str1:
            file = input_path + str1[1] + '.pgm'
            flag = mycopyfile(file, destination_path3)
            # num += 1
            # 统计开心人数
            if flag:
                funny += 1

# 按表情分
# # 完成提示
print('微笑%d人'% smiling)
print('严肃%d人'% serious)
print('开心%d人'% funny)
print('总共%d人'% (smiling+serious+funny))
print('完成')



