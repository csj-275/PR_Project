# 读取图像并显示
import numpy as np
import cv2

# 二进制读取文件
i = 4356
filename = './face/rawdata/' + str(i)
with open(filename, 'rb') as f:
    content = f.read()

# 将字节数组转成 128*128 的图像
data = np.frombuffer(content, dtype=np.uint8)

cv2.imshow('real',data.reshape(128, 128))
cv2.waitKey(0)


# img = Image.fromarray(data.reshape(128, 128), mode='L')
# print(img)
# 显示图像
# img.show()