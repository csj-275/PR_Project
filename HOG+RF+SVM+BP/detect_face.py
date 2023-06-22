
import cv2
from my_fun import get_img, non_image
import numpy as np
# 加载人脸检测器
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
num = []
if face_cascade.empty():
    print('加载分类器文件失败')
non_image = [1228, 1232, 1808, 4056, 4135, 4136, 5004]  # 没有数据特征的图片编号
# 读取需要检测的图片
size_image = [2412, 2416]

print(f'{len(non_image)}')
# 设置图像拼接输出大小
n_images = 4000-len(non_image)
cols = 60
rows = -(-n_images // cols)  # 向上取整
output_size = 16  # 输出图像大小
output = 255 * np.ones((output_size * rows, output_size * cols), dtype=np.uint8)
n = 0
for i in range(1223, 5223):
    if i in non_image:
        pass
    else:

        img = get_img(i)
        # 检测人脸
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.01, minNeighbors=2, minSize=(1, 1), flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        row = n // cols
        col = n % cols
        output[row * output_size:(row + 1) * output_size, col * output_size:(col + 1) * output_size] = cv2.resize(
            img, (output_size, output_size), interpolation=cv2.INTER_AREA)
        # # 判断是否检测到了人脸
        if len(faces) > 0:
            print('检测到了 {} 张人脸'.format(len(faces)))
        else:
            num.append(i)
            print('未检测到人脸')
        n += 1
print(num)
print(len(num))

# 显示输出图像
cv2.imshow('output', output)
cv2.imwrite('example_saved1.jpg', output)
cv2.waitKey()


# num = []
# for i in range(1223, 5223):
#     if i in non_image or i in size_image:
#         pass
#     else:
#         img = get_img(i)
#         print(sum(sum(np.array(img))))
#         if sum(sum(np.array(img))) == 0:
#             non_image.append(i)
#             print(f'=====图像{i}为空======')
#         if sum(sum(np.array(img))) > 18500:
#             non_image.append(i)
#             cv2.imshow(str(i), img)
#             cv2.waitKey()
# print(f'纯黑图的编号为：{non_image}')
# print(f'{len(non_image)}张图为空')

# 158 无效
# [1290, 1291, 1314, 1315, 1322, 1435, 1471, 1475, 1505, 1550, 1561, 1579, 1673, 1677, 1686, 1697, 1701, 1838, 1931, 2040, 2102, 2103, 2225, 2299, 2428, 2451, 2462, 2528, 2722, 2848, 3023, 3027, 3034, 3061, 3063, 3095, 3133, 3213, 3339, 3341, 3359, 3365, 3406, 3436, 3447, 3450, 3457, 3576, 3627, 3654, 3814, 3820, 3837, 3862, 3883, 4050, 4052, 4053, 4054, 4055, 4057, 4060, 4061, 4062, 4063, 4064, 4067, 4068, 4069, 4070, 4071, 4076, 4077, 4078, 4080, 4083, 4084, 4086, 4087, 4090, 4091, 4092, 4093, 4094, 4095, 4100, 4101, 4102, 4104, 4107, 4110, 4111, 4112, 4114, 4115, 4116, 4117, 4118, 4119, 4122, 4162, 4215, 4216, 4268, 4269, 4278, 4286, 4289, 4296, 4297, 4332, 4338, 4354, 4356, 4357, 4359, 4389, 4410, 4429, 4438, 4452, 4468, 4501, 4538, 4545, 4562, 4607, 4608, 4610, 4637, 4642, 4645, 4648, 4677, 4711, 4728, 4742, 4745, 4754, 4773, 4774, 4820, 4823, 4870, 4908, 4955, 4967, 4968, 4969, 5001, 5015, 5034, 5035, 5036, 5076, 5131, 5168, 5178]