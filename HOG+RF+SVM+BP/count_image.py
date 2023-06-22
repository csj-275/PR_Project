# 检测不存在的图像
import numpy as np
flag = []
i = 1223
while i <= 5222:
    try:
        filename = './face/rawdata/' + str(i)
        with open(filename, 'rb') as f:
            content = f.read()

    except:
        flag.append(i)
        print(f'第{i}张图不存在')
    i += 1
print(flag)