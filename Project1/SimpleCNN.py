import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import random_split, DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F


# Define the CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        # Dropout层
        self.dropout = nn.Dropout(0.5)

    # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # 展平(Flatten the tensor)
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def train_model():
    # 使用GPU加速
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Hyper-parameters(超参数)
    batch_size = 256
    # 验证集比例
    valid_size = 0.2
    num_epochs = 20
    # 进程数
    num_workers = 0

    # Data transformation with augmentation
    # 数据预处理
    transform = transforms.Compose([
        # 将图像转为数值
        # [0,255] → [0,1]
        transforms.ToTensor(),
        # [0,1] → [-1,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 数据集加载
    full_train_data = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform)

    # 分割训练集和验证集
    num_train = len(full_train_data)
    # store the test_dataset size as 20% of the total dataset
    split = int(np.floor(valid_size * num_train))
    train_size = num_train - split  # store the train_dataset size (80% in our case)

    train_data, valid_data = random_split(full_train_data, [train_size, split])

    # train the model using 80% of the dataset
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    # validate the working a validation dataset which contains 20% of the dataset
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    # run the test using the entire dataset
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    classes = ['plane', 'vehicle', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    net = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=0.001)
    optimizer = optim.SGD(net.parameters(), lr=0.03)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    valid_loss_min = np.Inf

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = 0.0
        valid_loss = 0.0

        # Training
        net.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            # 清梯度
            optimizer.zero_grad()
            # 前向传播
            output = net(data)
            # 计算损失
            loss = criterion(output, target)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        # Validation
        net.eval()
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = net(data)
                loss = criterion(output, target)
                valid_loss += loss.item() * data.size(0)

        # Calculate average loss
        train_loss /= len(train_loader.dataset)
        valid_loss /= len(valid_loader.dataset)

        end_time = time.time()
        epoch_time = end_time - start_time

        print(
            f'Epoch: {epoch + 1}/{num_epochs} '
            f'| Time: {epoch_time:.3f}s '
            f'| Training Loss: {train_loss:.4f} '
            f'| Validation Loss: {valid_loss:.4f}')

        # Save model if validation loss decreases
        if valid_loss <= valid_loss_min:
            print(
                f'Validation loss decreased ({valid_loss_min:.4f} --> {valid_loss:.4f}). '
                f'Saving model as net_cifar10.pt')
            torch.save(net.state_dict(), 'net_cifar10.pt')
            valid_loss_min = valid_loss

    # Load the best model
    net.load_state_dict(torch.load('net_cifar10.pt'))
    print('Finished Training')

    test_loss = 0.0
    class_correct = [0] * 10
    class_total = [0] * 10

    net.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            _, pred = torch.max(output, 1)
            correct = pred.eq(target.view_as(pred))
            for i in range(len(target)):
                label = target[i].item()
                class_correct[label] += correct[i].item()
                class_total[label] += 1
    # Print test results
    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.6f}')

    overall_accuracy = 100. * np.sum(class_correct) / np.sum(class_total)
    print(f'\nTest Accuracy (Overall): {overall_accuracy:.2f}%')


if __name__ == '__main__':
    train_model()
