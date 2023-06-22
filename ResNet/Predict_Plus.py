import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision
from PIL import Image
from glob import glob
from Print_Correct_Label import Print_Correct_Label
import random
import warnings


# bynum = [3391, 4982, 3153, 2088, 3530, 3898, 2447, 3725, 1380, 3815,
#          1342, 2960, 4238, 1700, 4659, 2206, 2511, 2664, 4939, 1974]

radom_times = 20
image_num = -1  # 指定对象输入, 随机时此处填-1
Class = 'PROP'
Version = 1
Epoch = 9
break_on = 0  # 遇错终止功能


break_flag = 0

# 忽略特定警告类型
warnings.filterwarnings("ignore", category=UserWarning)

# 获取所有样本号
all_src_file_list = glob('D:/python_pro/pgmdata/*')
src_labels = []
for i in range(len(all_src_file_list)):
    src_labels.append(all_src_file_list[i].split('\\')[-1])
    src_labels[i] = src_labels[i].split('.')[0]

# 获取类标签
src_file_list = glob('D:/python_pro/' + Class + '/ALL/*')
class_labels = []
for i in range(len(src_file_list)):
    class_labels.append(src_file_list[i].split('\\')[-1])

for kk in range(radom_times):
    if kk == 0:
        if image_num == -1:
            image_num = random.choice(src_labels)
            while ' ' not in Print_Correct_Label(image_num, Class, 0):
                image_num = random.choice(src_labels)
        else:
            break_flag = 1
    else:
        image_num = random.choice(src_labels)
        while ' ' not in Print_Correct_Label(image_num, Class, 0):
            image_num = random.choice(src_labels)

    # image_num = bynum[kk]

    # 加载预训练的ResNet模型
    model = torchvision.models.resnet18(pretrained=True)
    num_classes = len(class_labels)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 替换最后一层全连接层
    model.load_state_dict(torch.load(f'D:/大学/课程/模式识别/课程项目/人脸图像识别/python/MODEL/' + Class + '/' + Class +
                                     f'_ResNet18_V{Version}/ResNet18_V{Version}_Epoch{Epoch}.pth'))
    model.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),  # 将单通道图像转换为3通道
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用与训练时相同的均值和标准差
    ])

    # 加载待预测的图像
    image_path = f'D:/python_pro/pgmdata/{image_num}.pgm'
    image = Image.open(image_path)
    input_image = transform(image).unsqueeze(0)  # 在维度0上添加一个批次维度

    # 预测图像类别
    if Class == 'SEX':
        with torch.no_grad():
            output = model(input_image)
            probabilities = torch.sigmoid(output)
            _, predicted_labels = probabilities.topk(2)  # 获取预测概率最高的2个类别
    else:
        with torch.no_grad():
            output = model(input_image)
            probabilities = torch.sigmoid(output)
            _, predicted_labels = probabilities.topk(3)  # 获取预测概率最高的3个类别

    correct_label = Print_Correct_Label(image_num, Class, 0)

    # 打印预测结果
    print(f'\033[1m\033[33m\033[3mLabel_{kk + 1}:\t{image_num}\033[0m', end='\t\t\t\t\t\t\t')
    if probabilities[0, predicted_labels[0, 1]].item() < 0.91:
        # print(probabilities[0, predicted_labels[0, 1]].item())
        if class_labels[predicted_labels[0, 0]] == correct_label:
            print('\033[1m\033[32mCorrect\033[0m')
            for i in range(predicted_labels.size(1)):
                class_index = predicted_labels[0, i]
                class_label = class_labels[class_index]
                probability = probabilities[0, class_index]
                print(f"Predicted label: {class_label}, Probability: {probability.item():.4f}")
        else:
            print('\033[1m\033[31mWrong\033[0m')
            for i in range(predicted_labels.size(1)):
                class_index = predicted_labels[0, i]
                class_label = class_labels[class_index]
                probability = probabilities[0, class_index]
                print(f"\033[41mPredicted label: {class_label}, Probability: {probability.item():.4f}\033[0m")
            if break_on:
                break_flag = 1
    else:
        if (class_labels[predicted_labels[0, 0]] in correct_label) and (class_labels[predicted_labels[0, 1]] in correct_label):
            print('\033[1m\033[32mCorrect\033[0m')
            for i in range(predicted_labels.size(1)):
                class_index = predicted_labels[0, i]
                class_label = class_labels[class_index]
                probability = probabilities[0, class_index]
                print(f"Predicted label: {class_label}, Probability: {probability.item():.4f}")
        else:
            print('\033[1m\033[31mWrong\033[0m')
            for i in range(predicted_labels.size(1)):
                class_index = predicted_labels[0, i]
                class_label = class_labels[class_index]
                probability = probabilities[0, class_index]
                print(f"\033[41mPredicted label: {class_label}, Probability: {probability.item():.4f}\033[0m")
            if break_on:
                break_flag = 1


    print(f'Correct label: {correct_label}')

    if break_flag:
        break
