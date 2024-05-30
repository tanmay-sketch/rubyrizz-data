import torch
import torch.nn as nn

class YOLOv5Simple(nn.Module):
    def __init__(self, num_classes=6):
        super(YOLOv5Simple, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * 20 * 20, 1024)
        self.fc2 = nn.Linear(1024, self.num_classes * 5)  # (class_score, x, y, w, h)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.conv3(x))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.conv4(x))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.conv5(x))
        x = nn.MaxPool2d(2)(x)
        x = x.view(-1, 256 * 20 * 20)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, self.num_classes, 5)
        return x
