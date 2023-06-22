# 存储常用函数
import re
from skimage.feature import hog
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix # 计算得分
non_image = [1228, 1232, 1808, 2099, 2100, 2101, 2102, 2103, 2104, 2105, 2106, 3283, 3860, 3861, 3862, 3883, 3965, 4056, 4125, 4135, 4136, 4146, 4237, 4267, 4295, 4335, 4354, 4355, 4356, 4357, 4358, 4359, 4429, 4452, 4498, 4566, 4637, 4679, 4710, 4779, 4908, 4992, 5004, 5076, 5113]
# non_image = [1228, 1232, 1808, 4056, 4135, 4136, 5004]
#non_image = non_image + [1236, 1254, 1256, 1290, 1291, 1314, 1315, 1322, 1361, 1372, 1432, 1435, 1444, 1446, 1460, 1471, 1475, 1493, 1497, 1499, 1501, 1504, 1505, 1507, 1516, 1542, 1550, 1561, 1579, 1642, 1646, 1647, 1656, 1666, 1668, 1673, 1677, 1686, 1689, 1697, 1701, 1705, 1706, 1729, 1732, 1738, 1790, 1794, 1800, 1802, 1822, 1838, 1840, 1842, 1865, 1875, 1883, 1895, 1901, 1920, 1931, 1938, 1947, 2039, 2040, 2099, 2100, 2101, 2102, 2103, 2104, 2105, 2106, 2149, 2151, 2166, 2173, 2199, 2219, 2220, 2221, 2225, 2235, 2240, 2287, 2288, 2290, 2297, 2299, 2318, 2329, 2366, 2368, 2370, 2374, 2406, 2428, 2429, 2444, 2449, 2451, 2452, 2456, 2459, 2462, 2501, 2506, 2517, 2528, 2535, 2641, 2653, 2654, 2669, 2673, 2703, 2711, 2722, 2725, 2727, 2765, 2822, 2848, 2849, 2880, 2924, 2930, 2941, 2957, 2969, 2996, 3019, 3023, 3027, 3031, 3034, 3037, 3038, 3061, 3062, 3063, 3064, 3095, 3123, 3133, 3158, 3160, 3190, 3213, 3217, 3231, 3239, 3247, 3282, 3283, 3333, 3337, 3339, 3341, 3343, 3359, 3364, 3365, 3406, 3422, 3427, 3431, 3434, 3436, 3437, 3438, 3444, 3446, 3447, 3448, 3449, 3450, 3452, 3454, 3457, 3576, 3592, 3600, 3617, 3627, 3654, 3663, 3665, 3677, 3785, 3786, 3788, 3803, 3805, 3814, 3818, 3820, 3823, 3837, 3860, 3861, 3862, 3883, 4033, 4050, 4051, 4052, 4053, 4054, 4055, 4057, 4058, 4060, 4061, 4062, 4063, 4064, 4065, 4067, 4068, 4069, 4070, 4071, 4074, 4076, 4077, 4078, 4080, 4083, 4084, 4085, 4086, 4087, 4090, 4091, 4092, 4093, 4094, 4095, 4096, 4100, 4101, 4102, 4103, 4104, 4107, 4110, 4111, 4112, 4114, 4115, 4116, 4117, 4118, 4119, 4122, 4125, 4146, 4162, 4163, 4215, 4216, 4224, 4226, 4227, 4228, 4237, 4249, 4254, 4256, 4267, 4268, 4269, 4275, 4278, 4279, 4286, 4289, 4295, 4296, 4297, 4298, 4322, 4323, 4324, 4327, 4332, 4335, 4338, 4354, 4355, 4356, 4357, 4358, 4359, 4364, 4386, 4389, 4400, 4401, 4410, 4429, 4437, 4438, 4450, 4452, 4456, 4464, 4468, 4491, 4493, 4498, 4501, 4529, 4538, 4545, 4551, 4556, 4562, 4566, 4572, 4587, 4607, 4608, 4610, 4611, 4614, 4618, 4628, 4637, 4642, 4643, 4645, 4648, 4676, 4677, 4678, 4679, 4681, 4683, 4689, 4710, 4711, 4712, 4714, 4717, 4728, 4731, 4735, 4737, 4739, 4742, 4743, 4745, 4746, 4749, 4754, 4762, 4767, 4773, 4774, 4775, 4779, 4790, 4791, 4820, 4823, 4824, 4870, 4875, 4879, 4908, 4910, 4913, 4915, 4924, 4925, 4928, 4955, 4967, 4968, 4969, 4982, 4992, 4995, 4997, 5001, 5005, 5006, 5015, 5016, 5018, 5024, 5029, 5034, 5035, 5036, 5043, 5047, 5058, 5059, 5060, 5073, 5076, 5113, 5131, 5136, 5145, 5147, 5159, 5168, 5174, 5177, 5178, 5181, 5209, 5217, 5221, 5222]

def get_img(i):
    filename = './face/rawdata/' + str(i)
    with open(filename, 'rb') as f:
        content = f.read()
    data = np.frombuffer(content, dtype=np.uint8)
    if i in [2412, 2416]:
        img = data.reshape(512, 512)
    else:
        img = data.reshape(128, 128)
    return img

def extract_description(filename, all): # 提取特征描述
    my_image = [str(i) for i in non_image]
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            pattern = r'\((.*?)\s+(.*?)\)'  # 匹配形式
            items = re.findall(pattern, line)
            d = {k.strip('_'): v.strip("'") for k, v in items}  # 将特征描述存入字典
            nums = re.findall(r'\d+', line)  # 找出数字
            nums = ''.join(nums)
            if nums not in my_image:
                d['prop'] = d['prop'] + ')'
                all[nums] = d
                # print(f'{nums}: {all[nums]}')  # 打印字典
            line = f.readline()
    return all

def show_mis_img(y_pre, y, X): # 获取分类错误的图像
    misclassified_indices = np.nonzero(y_pre != y)[0]
    # 随机选择一些错误的图像进行显示
    num_to_display = 5
    sample_indices = np.random.choice(misclassified_indices, num_to_display, replace=False)
    # 显示分类错误的图像
    for i, index in enumerate(sample_indices):
        plt.subplot(1, num_to_display, i + 1)
        plt.imshow(X[index], cmap='gray')
        plt.title("Predicted: {}, True: {}".format(y_pre[index], y[index]))
        plt.axis('off')
    plt.show()

def hog_improve(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), visualize=False, multichannel=True, feature_vector=True, scales=[0.75, 1, 1.25]):
    hogs = []
    for scale in scales:
        # 缩放图片
        resized_img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
        # HOG特征提取
        hog_features = hog(resized_img, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=visualize)
        hogs.append(hog_features)
    # 将多个尺度的特征拼接在一起
    features = np.concatenate(hogs)
    return features.tolist()

def hog_before(X):
    image_descriptors_single = []
    fd, _ = hog(X, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(8, 8),
                            block_norm='L2-Hys', visualize=True)
    return fd.tolist()

def print_result(Y_test, Y_predict, method): # 打印结果
    acc = accuracy_score(Y_test, Y_predict)
    precision = precision_score(Y_test, Y_predict, average='macro', zero_division = 0)
    recall = recall_score(Y_test, Y_predict, average='macro')
    cm = confusion_matrix(Y_test, Y_predict)
    print('=============='+method+'==============')
    print('confusion matrix:\n', cm)
    print('acc: ', acc)
    print('precision: ', precision)
    print('recall: ', recall)
    return acc, precision, recall