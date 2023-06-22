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
destination_path1 = './race/white/'
destination_path2 = './race/black/'
destination_path3 = './race/asian/'
destination_path4 = './race/hispanic/'
destination_path5 = './race/other/'

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

white = 0
black = 0
asian = 0
hispanic = 0
other = 0
# 一行行读入标签文件
label_files = [label_file1,label_file2]
for file in label_files:
    label_file = open(file,'r')
    data = label_file.readlines()
    for line in data:
        str1 = line.split(' ')
        # print(str1[1])
        if 'white)' in str1:
            file = input_path + str1[1] + '.pgm'
            flag = mycopyfile(file,destination_path1)
            # num += 1
            # 统计白人人数
            if flag:
                white += 1
        elif 'black)' in str1:
            file = input_path + str1[1] + '.pgm'
            flag = mycopyfile(file, destination_path2)
            # num += 1
            # 统计黑人人数
            if flag:
                black += 1
        elif 'asian)' in str1:
            file = input_path + str1[1] + '.pgm'
            flag = mycopyfile(file, destination_path3)
            # num += 1
            # 统计黄种人人数
            if flag:
                asian += 1
        elif 'hispanic)' in str1:
            file = input_path + str1[1] + '.pgm'
            flag = mycopyfile(file, destination_path4)
            # num += 1
            # 统计拉丁美洲人人数
            if flag:
                hispanic += 1
        elif 'other)' in str1:
            file = input_path + str1[1] + '.pgm'
            flag = mycopyfile(file, destination_path5)
            # num += 1
            # 统计拉丁美洲人人数
            if flag:
                other += 1


# 按表情分
# # 完成提示
print('白种人%d人'% white)
print('黑种人%d人'% black)
print('黄种人%d人'% asian)
print('拉丁美洲人%d人'% hispanic)
print('其他%d人'% other)
print('总共%d人'% (white+black+asian+hispanic+other))
print('完成')



