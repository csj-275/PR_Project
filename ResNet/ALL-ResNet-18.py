import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import multiprocessing
import os
import Journal

Class = 'SEX'
Version = '4'
seed = 0  # 42 123
t_num = 0.8
batch_size = 32  # 32
lr = 0.001  #0.001
momentum = 0.9  # 0.9
layer = 18
num_epochs = 30

if __name__ == '__main__':
    # 在这里添加 freeze_support()
    multiprocessing.freeze_support()

    if not os.path.exists('D:/大学/课程/模式识别/课程项目/人脸图像识别/python/MODEL/' + Class + '/'):
        os.mkdir(r'D:/大学/课程/模式识别/课程项目/人脸图像识别/python/MODEL/' + Class + '/')
    if not os.path.exists('D:/大学/课程/模式识别/课程项目/人脸图像识别/python/MODEL/' + Class + '/' + Class +
                          '_ResNet18_V' + Version):
        os.mkdir(r'D:/大学/课程/模式识别/课程项目/人脸图像识别/python/MODEL/' + Class + '/' + Class + '_ResNet18_V' + Version)
    journal = Journal.Journal(Class, Version, seed, t_num, batch_size, lr, momentum, layer)

    # 设置随机种子以保持结果的一致性
    torch.manual_seed(seed)

    # 定义训练批次大小和设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print('使用GPU')

    # 数据预处理和增强
    print('数据预处理中........')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    # 加载数据集并应用预处理,创建数据加载器
    train_dataset = datasets.ImageFolder('D:/python_pro/' + Class + '/train_data3', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    test_dataset = datasets.ImageFolder('D:/python_pro/' + Class + '/test_data3', transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    # 加载ResNet-18模型
    print('加载预训练ResNet-18模型........')
    model = models.resnet18(pretrained=True)
    num_classes = len(train_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 替换最后一层全连接层

    # 将模型移动到设备上
    print('将模型移动到设备上........')
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr, momentum)   # 0.001, 0.9

    # 训练模型
    TL = 100
    TR = 0
    for epoch in range(num_epochs):
        print('模型训练中........')
        running_loss = 0.0
        train_correct = 0
        train_accuracy = 0.0
        total = 0

        model.train()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # # 前向传播
            # outputs = model(images)
            # loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)

            running_loss += loss.item()
            total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_accuracy = 100 * train_correct / total

        # 输出每个epoch的训练损失
        epoch_loss = running_loss / len(train_loader)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Accuracy on train set: {train_accuracy:.2f}%, Training Loss: {epoch_loss:.4f}")
        journal.write_journal(f"Epoch [{epoch + 1}/{num_epochs}], Accuracy on test set: {train_accuracy:.2f}%, "
                              f"Training Loss: {epoch_loss:.4f}\n")

        # 在测试集上评估模型
        print('模型评估中........')
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # 输出模型在测试集上的准确率
        accuracy = 100 * correct / total
        print(f"Accuracy on test set: {accuracy:.2f}%")
        journal.write_journal(f"Accuracy on test set: {accuracy:.2f}%\n\n")
        # if epoch_loss < TL or accuracy > TR:
        #     print('正在保存模型........')
        #     torch.save(model.state_dict(), 'D:/大学/课程/模式识别/课程项目/人脸图像识别/python/MODEL/' + Class + '/' + Class +
        #                '_ResNet18_V' + Version + '/ResNet18_V' + Version + '_Epoch' + str(epoch + 1) + '.pth')
        #     TL, TR = epoch_loss, accuracy
        if accuracy > 97:
            print('正在保存模型........')
            torch.save(model.state_dict(), 'D:/大学/课程/模式识别/课程项目/人脸图像识别/python/MODEL/' + Class + '/' + Class +
                       '_ResNet18_V' + Version + '/ResNet18_V' + Version + '_Epoch' + str(epoch + 1) + '.pth')
