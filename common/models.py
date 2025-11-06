import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class LeNet(nn.Module):
    def __init__(self, num_classes=10, out_feat=False):
        super().__init__()
        self.out_feat = out_feat
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        feat = F.relu(self.fc2(x))
        logits = self.fc3(feat)
        if self.out_feat:
            return logits, feat
        return logits

def small_cnn_cifar(num_classes=10, out_feat=False):
    m = resnet18(num_classes=num_classes)
    if out_feat:
        # Adapt to return penultimate features
        class Wrap(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.backbone = nn.Sequential(*(list(m.children())[:-1]))
                self.fc = m.fc
            def forward(self, x):
                x = self.backbone(x).flatten(1)
                feat = x
                logits = self.fc(x)
                return logits, feat
        return Wrap(m)
    return m
