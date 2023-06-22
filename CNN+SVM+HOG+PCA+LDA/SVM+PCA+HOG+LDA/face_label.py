import os
import pandas as pd
import shutil

# path = 'D:/2023PR/face/sex1/female'
path = 'D:/2023PR/face/sex_copy/female'
path1 = 'D:/2023PR/face/sex_copy/male'
final_path = 'D:/2023PR/face/newsex'
female_count = 0

files = [i for i in os.listdir(path)]  # os.listdir返回指定目录下的所有文件和目录名
for file, i in zip(files, range(len(files))):
    old = path + '/' + file
    new = path + '/' + '0'+str(i)+'.pgm'
    print(old)
    print(new)
    os.rename(old, new)
    shutil.move(new, final_path)
    female_count += 1
print(female_count)

male_files = [j for j in os.listdir(path1)]  # os.listdir返回指定目录下的所有文件和目录名
for male_file, j in zip(male_files, range(female_count, female_count+len(male_files))):
    male_old = path1 + '/' + male_file
    male_new = path1 + '/' + '0' + str(j)+'.pgm'
    print(male_old)
    print(male_new)
    os.rename(male_old, male_new)
    shutil.move(male_new, final_path)
