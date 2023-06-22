# 读取数据
import re
import numpy as np

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

def main():
    label_dict = dict()  # 存放所有数据特征
    label_dict = extract_description('./face/faceDR', label_dict)
    label_dict = extract_description('./face/faceDS', label_dict)
    img_num_list = label_dict.keys() # 获取所有有标签的图像
    img_dict = dict()
    for i in img_num_list:
        filename = './face/rawdata/' + i # 图像文件路径
        with open(filename, 'rb') as f:
            content = f.read()
            img_dict[i] = content

    print('读取数据完成')


if __name__ == '__main__':
    main()