import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import Journal
import os

Class = 'SEX'
Version = '4'
seed = 0  # 42 123
t_num = 0.7
batch_size = 32  # 32
lr = 0.001  # 0.001
momentum = 0.9  # 0.9
save_model = 0
layer = 50

if not os.path.exists('D:/大学/课程/模式识别/课程项目/人脸图像识别/python/MODEL/' + Class + '/'):
    os.mkdir(r'D:/大学/课程/模式识别/课程项目/人脸图像识别/python/MODEL/' + Class + '/')
if not os.path.exists('D:/大学/课程/模式识别/课程项目/人脸图像识别/python/MODEL/' + Class + '/' + Class +
                      '_ResNet50_V' + Version):
    os.mkdir(r'D:/大学/课程/模式识别/课程项目/人脸图像识别/python/MODEL/' + Class + '/' + Class + '_ResNet50_V' + Version)
journal = Journal.Journal(Class, Version, seed, t_num, batch_size, lr, momentum, layer)

# 设置随机种子以保持结果的一致性
torch.manual_seed(seed)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('使用GPU')

# 数据预处理和增强
print('数据预处理中........')
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
])

# 加载数据集并应用预处理
train_dataset = datasets.ImageFolder('D:/python_pro/SEX/train_data', transform=transform)
test_dataset = datasets.ImageFolder('D:/python_pro/SEX/test_data', transform=transform)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载预训练的ResNet模型
print('加载预训练的ResNet模型........')
model = models.resnet50(pretrained=True)

# 冻结预训练的模型参数
print('冻结预训练的模型参数........')
for param in model.parameters():
    param.requires_grad = False

# 替换最后一层全连接层
num_classes = 2  # 两个类别：男性和女性
model.fc = nn.Linear(2048, num_classes)

# 将模型移动到设备上
print('将模型移动到设备上........')
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 训练模型
num_epochs = 5000
TL = 100
TR = 0
train_loss = 10000
for epoch in range(num_epochs):
    if train_loss < 0.2:
        lr = 0.0005
    elif train_loss < 0.1:
        lr = 0.0001
    print('模型训练中........')
    train_loss = 0.0
    train_correct = 0
    total = 0

    model.train()

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计训练损失和准确率
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss /= total
    train_accuracy = 100 * train_correct / total

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    journal.write_journal(f"Epoch [{epoch + 1}/{num_epochs}], Accuracy on test set: {train_accuracy:.2f}%, "
                          f"Training Loss: {train_loss:.4f}\n")

    # 在测试集上评估模型
    print('模型评估中........')
    model.eval()

    test_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            test_correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_accuracy = 100 * test_correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    journal.write_journal(f"Accuracy on test set: {test_accuracy:.2f}%\n\n")

    if train_loss < 0.007:
        torch.save(model.state_dict(), 'D:/大学/课程/模式识别/课程项目/人脸图像识别/python/MODEL/' + Class + '/' + Class +
                      '_ResNet50_V' + Version + '/ResNet50_V' + Version + 'Epoch' + str(epoch + 1) + '.pth')
        break

    if (train_loss < TL or train_accuracy > TR) and save_model:
        print('正在保存模型........')
        torch.save(model.state_dict(), 'ResNet50_V2_Epoch' + str(epoch + 1) + '.pth')
        TL, TR = train_loss, train_accuracy

# 保存训练好的模型
# print('正在保存模型........')
# torch.save(model.state_dict(), 'gender_classification_model.pth')
