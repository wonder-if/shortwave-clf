from torch import nn
import torch

__all__ = ["ResNet"]


class ResidualUnit(nn.Module):
    def __init__(self, channels, down=False):
        super(ResidualUnit, self).__init__()
        stride = 2 if down else 1
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
        )
        if down:
            self.down_sampler = nn.Conv1d(
                channels, channels, kernel_size=1, stride=stride
            )
        else:
            self.down_sampler = nn.Identity()

    def forward(self, x):
        return self.down_sampler(x) + self.layers(x)


class ResidualStack(nn.Module):
    def __init__(self, channels):
        super(ResidualStack, self).__init__()
        self.layers = nn.Sequential(
            ResidualUnit(channels, down=True),
            # ResidualUnit(channels),
        )

    def forward(self, x):
        return self.layers(x)


class ResNet(nn.Module):
    def __init__(
        self, num_classes=5, channels=32
    ):  # 假设二分类问题，可根据实际需求修改
        super(ResNet, self).__init__()
        self.block1 = nn.Sequential(nn.Conv1d(1, channels, 7, 1, 3))  # 输入通道数改为1
        self.convs = nn.Sequential(
            ResidualStack(channels),
            ResidualStack(channels),
            ResidualStack(channels),
            ResidualStack(channels),
            ResidualStack(channels),
            ResidualStack(channels),
            ResidualStack(channels),
            ResidualStack(channels),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(channels, num_classes)  # 修改分类头的输出类别数

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.convs(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.proj(x)
        return x
