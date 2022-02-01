"""
    File name: CIFAR_10_Model.py
    Description: An easy implement of LeNet Model

    Author: Botian Lan
    Time: 2022/01/22
"""
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear
from layers._Flatten import _Flatten


# Models class
class LeNet(nn.Module):
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

        # Batch Norm
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(10)

        # Relu
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.Conv1(x)
        x = self.relu(self.bn1(x))
        x = self.Maxpool1(x)
        x = self.Conv2(x)
        x = self.relu(self.bn2(x))
        x = self.Maxpool2(x)
        x = self.Conv3(x)
        x = self.relu(self.bn3(x))
        x = self.Maxpool3(x)
        x = self.Flatten(x)
        x = self.Linear1(x)
        x = self.relu(self.bn4(x))
        x = self.Linear2(x)
        outputs = self.relu(self.bn5(x))
        return outputs


if __name__ == '__main__':
    import torch
    x = torch.randn([4, 3, 32, 32])
    model = LeNet()


    def gener(model):
        for para in model.parameters():
            yield para

    for groups in gener(model):
        print(groups.shape)

