import torch
import torch.nn as nn


def get_norm(norm_type, num_channels):
    if norm_type == 'batch':
        return nn.BatchNorm2d(num_channels)
    if norm_type == 'group':
        # use 8 groups (fallback to 1 if channels < 8)
        groups = min(8, num_channels)
        return nn.GroupNorm(groups, num_channels)
    if norm_type == 'instance':
        return nn.InstanceNorm2d(num_channels)
    return nn.Identity()


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm='batch', dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            get_norm(norm, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            get_norm(norm, out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.drop(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, num_classes=10, base_channels=32, norm='batch', dropout=0.0):
        super().__init__()
        self.layer1 = ConvBlock(3, base_channels, norm=norm, dropout=dropout)
        self.layer2 = ConvBlock(base_channels, base_channels * 2, norm=norm, dropout=dropout)
        self.layer3 = ConvBlock(base_channels * 2, base_channels * 4, norm=norm, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.classifier(x)
        return x


def build_model(num_classes=10, dropout=0.3, norm='batch', device=None, base_channels=32):
    model = ConvNet(num_classes=num_classes, dropout=dropout, norm=norm, base_channels=base_channels)
    if device is not None:
        model = model.to(device)
    return model
