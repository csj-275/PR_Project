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
destination_path1 = './age/child/'
destination_path2 = './age/adult/'
destination_path3 = './age/teen/'
destination_path4 = './age/senior'

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

child = 0
adult = 0
teen = 0
senior = 0
# 一行行读入标签文件
label_files = [label_file1,label_file2]
for file in label_files:
    label_file = open(file,'r')
    data = label_file.readlines()
    for line in data:
        str1 = line.split(' ')
        # print(str1[1])
        if 'child)' in str1:
            file = input_path + str1[1] + '.pgm'
            flag = mycopyfile(file,destination_path1)
            # num += 1
            # 统计孩子人数
            if flag:
                child += 1
        elif 'adult)' in str1:
            file = input_path + str1[1] + '.pgm'
            flag = mycopyfile(file, destination_path2)
            # num += 1
            # 统计成人人数
            if flag:
                adult += 1
        elif 'teen)' in str1:
            file = input_path + str1[1] + '.pgm'
            flag = mycopyfile(file,destination_path3)
            # num += 1
            # 统计青年人数
            if flag:
                teen += 1
        elif 'senior)' in str1:
            file = input_path + str1[1] + '.pgm'
            flag = mycopyfile(file,destination_path4)
            # num += 1
            # 统计老人人数
            if flag:
                senior += 1

# 按年龄分
# # 完成提示
print('小孩%d人'% child)
print('青年%d人'% teen)
print('成人%d人'% adult)
print('老人%d人'% senior)
print('总共%d人'% (child+teen+adult+senior))
print('完成')



