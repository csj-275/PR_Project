import numpy as np
import cv2
from my_fun import get_img, non_image
num = []
# 读取需要检测的图片
size_image = [2412, 2416]
for i in range(1223, 5223):
    if i in non_image or i in size_image:
        pass
    else:
        img = get_img(i)
        print(sum(sum(np.array(img))))
        if sum(sum(np.array(img))) > 15000:
            print(f'当前为图像{i}')
            non_image.append(i)
            cv2.imshow(str(i), img)
            cv2.waitKey()
print(f'纯黑图的编号为：{non_image}')
print(f'{len(non_image)}张图为空')