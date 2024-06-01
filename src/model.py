import torch
import torch.nn as nn
import torch.nn.init as init

class YOLOv5Moderate(nn.Module):
    def __init__(self, num_classes=6):
        super(YOLOv5Moderate, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512 * 20 * 20, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, self.num_classes * 5)  # (class_score, x, y, w, h)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.bn3(self.conv3(x)))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.bn4(self.conv4(x)))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.bn5(self.conv5(x)))
        x = nn.MaxPool2d(2)(x)
        x = x.view(-1, 512 * 20 * 20)
        x = nn.ReLU()(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        x = x.view(-1, self.num_classes, 5)
        return x
