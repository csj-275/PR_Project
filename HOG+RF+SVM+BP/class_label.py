# 标签编码
import re
import numpy as np
def extract_description(filename, all): # 提取特征描述
    non_image = [1228, 1232, 1808, 4056, 4135, 4136, 5004, 2099, 2100, 2101, 2102, 2103, 2104, 2105, 2106, 3283, 3860,
                 3861, 3862, 3883, 4125, 4146,
                 4237, 4267, 4295, 4335, 4354, 4355, 4358, 4429, 4452, 4498, 4566, 4637, 4679, 4710, 4779, 4908, 4992,
                 5076, 5113]
    non_image = [str(i) for i in non_image]
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            pattern = r'\((.*?)\s+(.*?)\)'  # 匹配形式
            items = re.findall(pattern, line)
            d = {k.strip('_'): v.strip("'") for k, v in items}  # 将特征描述存入字典
            nums = re.findall(r'\d+', line)  # 找出数字
            nums = ''.join(nums)
            if nums not in non_image:
                d['prop'] = d['prop'] + ')'
                all[nums] = d
                # print(f'{nums}: {all[nums]}')  # 打印字典
            line = f.readline()
    return all


all = dict() # 存放所有数据特征
all = extract_description('./face/faceDR', all)
all = extract_description('./face/faceDS', all)

sex = []

for key in all.keys():
    if all[key]['sex'] == 'male':
        sex.append(1)
    else:
        sex.append(0)

Y = np.array(sex)
print(Y)