# 正则表达式提取标签
import re

def extract_description(filename, all): # 提取特征描述
    non_image = [1228, 1232, 1808, 4056, 4135, 4136, 5004]  # 没有数据特征的图片编号
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
                print(f'{nums}: {all[nums]}')  # 打印字典
            line = f.readline()
    return all


all = dict() # 存放所有数据特征
all = extract_description('./face/faceDR', all)
all = extract_description('./face/faceDS', all)


