"""
    File name: CIFAR_10_Model.py
    Description: An easy implement of CIFAR-10 Model

    Author: Botian Lan
    Time: 2022/01/22
"""
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear
from layers._Flatten import _Flatten


# Models class
class CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolution layers
        self.Conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.Conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.Conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)

        # Max pooling layers
        self.Maxpool1 = MaxPool2d(kernel_size=2)
        self.Maxpool2 = MaxPool2d(kernel_size=2)
        self.Maxpool3 = MaxPool2d(kernel_size=2)

        # Fully connected layers
        self.Flatten = _Flatten()
        self.Linear1 = Linear(in_features=1024, out_features=64)
        self.Linear2 = Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Maxpool1(x)
        x = self.Conv2(x)
        x = self.Maxpool2(x)
        x = self.Conv3(x)
        x = self.Maxpool3(x)
        x = self.Flatten(x)
        x = self.Linear1(x)
        outputs = self.Linear2(x)
        return outputs


if __name__ == '__main__':
    import torch
    x = torch.randn([4, 3, 32, 32])
    model = CIFAR10()
    print(model(x))