import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import random_split, DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F


# ============================
# Residual Block
# ============================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ============================
# ResNet Model
# ============================
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for s in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ============================
# Training
# ============================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 128
    valid_size = 0.2
    num_epochs = 30
    num_workers = 0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    full_train_data = torchvision.datasets.CIFAR10(
        'data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(
        'data', train=False, download=True, transform=transform)

    split = int(len(full_train_data) * valid_size)
    train_size = len(full_train_data) - split
    train_data, valid_data = random_split(
        full_train_data, [train_size, split])

    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_data, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                             shuffle=False)

    net = ResNet().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    best_valid = np.Inf

    for epoch in range(num_epochs):
        start = time.time()
        train_loss = 0
        valid_loss = 0

        # ---- train ----
        net.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = net(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        # ---- validate ----
        net.eval()
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                out = net(x)
                loss = criterion(out, y)
                valid_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)
        valid_loss /= len(valid_loader.dataset)

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train {train_loss:.4f} | "
              f"Valid {valid_loss:.4f} | "
              f"Time {time.time() - start:.1f}s")

        if valid_loss < best_valid:
            torch.save(net.state_dict(), "resnet_cifar10.pt")
            best_valid = valid_loss
            print("Saved Best Model")

    # ============================
    # Test
    # ============================
    net.load_state_dict(torch.load("resnet_cifar10.pt"))
    net.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = net(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    train_model()
