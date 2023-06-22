# 标签编码
from my_fun import non_image, extract_description
size_image = [2412, 2416]


all = dict() # 存放所有数据标签
all = extract_description('./face/faceDR', all)
all = extract_description('./face/faceDS', all)
age = []

label = dict()
label['sex'] = ['male', 'female']
label['age'] = ['senior', 'adult', 'teen', 'child']
label['race'] = ['white', 'hispanic', 'asian', 'black', 'other']
label['face'] = ['smiling', 'serious', 'funny']

age_list, sex_list, race_list, face_list = [],[],[],[]

for key in all.keys(): # 遍历提取完的标签字典
    if int(key) in non_image or int(key) in size_image:
        pass
    else:
        age = all[key]['age']
        sex = all[key]['sex']
        face = all[key]['face']
        race = all[key]['race']
        age_list.append(label['age'].index(age))
        sex_list.append(label['sex'].index(sex))
        face_list.append(label['face'].index(face))
        race_list.append(label['race'].index(race))

