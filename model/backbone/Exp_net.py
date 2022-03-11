import numpy as np
import torch
from torch.nn import BatchNorm2d, BatchNorm1d, Conv2d, Linear, MaxPool2d, UpsamplingNearest2d, Dropout2d
from torch import nn
from layers import Flatten


class ExpNet(nn.Module):
    def __init__(self, batch_size):
        super(ExpNet, self).__init__()
        self.batch_size = batch_size
        # Conv layer
        self.Conv1 = Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.Conv2 = Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.Conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.Conv4 = Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.Conv5 = Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1)

        # Max pooling
        self.Maxpool1 = MaxPool2d(kernel_size=(2,2), stride=2)
        self.Maxpool2 = MaxPool2d(kernel_size=(2,2), stride=2)

        # Nearest Up sampling
        self.NNI1 = UpsamplingNearest2d(scale_factor=2)
        self.NNI2 = UpsamplingNearest2d(scale_factor=2)

        # Dropout layer
        self.Drop = Dropout2d(p=0.5)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Drop(x)
        x = self.Maxpool1(x)
        x = self.Conv2(x)
        x = self.Drop(x)
        x = self.Maxpool2(x)
        x = self.Conv3(x)
        x = self.Drop(x)

        x = self.NNI1(x)
        x = self.Conv4(x)
        x = self.Drop(x)
        x = self.NNI2(x)
        x = self.Conv5(x)
        x = self.Drop(x)
        x = torch.sum(x, dim=1)
        x = torch.unsqueeze(x, dim=1)
        return x


if __name__ == '__main__':
    import torch.nn.functional as F
    alex = ExpNet(batch_size=1)
    x = torch.randn([2, 1, 256, 256])
    x = alex(x)
    print(x.shape)
