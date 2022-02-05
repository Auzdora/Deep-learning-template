import torch
from torch.nn import BatchNorm2d, BatchNorm1d, Conv2d, Linear, MaxPool2d
from torch import nn
from layers._Flatten import _Flatten


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # Conv layer
        self.Conv1 = Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=(4, 4))
        self.Conv2 = Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding=2)
        self.Conv3 = Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.Conv4 = Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.Conv5 = Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1)

        # Max pool
        self.Maxpool = MaxPool2d(kernel_size=(3, 3), stride=2)

        # Linear layer
        self.Flatten = _Flatten()
        self.Linear1 = Linear(in_features=9216, out_features=4096)
        self.Linear2 = Linear(in_features=4096, out_features=4096)
        self.Linear3 = Linear(in_features=4096, out_features=1000)
        self.Linear4 = Linear(in_features=1000, out_features=10)

        # relu
        self.Relu = nn.ReLU()

        # Batch Norm
        self.bn1 = BatchNorm2d(96)
        self.bn2 = BatchNorm2d(256)
        self.bn3 = BatchNorm2d(384)
        self.bn4 = BatchNorm2d(384)
        self.bn5 = BatchNorm2d(256)
        self.bn6 = BatchNorm1d(4096)
        self.bn7 = BatchNorm1d(4096)
        self.bn8 = BatchNorm1d(1000)
        self.bn9 = BatchNorm1d(10)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Relu(self.bn1(x))
        x = self.Maxpool(x)
        x = self.Conv2(x)
        x = self.Relu(self.bn2(x))
        x = self.Maxpool(x)
        x = self.Conv3(x)
        x = self.Relu(self.bn3(x))
        x = self.Conv4(x)
        x = self.Relu(self.bn4(x))
        x = self.Conv5(x)
        x = self.Relu(self.bn5(x))
        x = self.Maxpool(x)
        x = self.Flatten(x)
        x = self.Linear1(x)
        x = self.Relu(self.bn6(x))
        x = self.Linear2(x)
        x = self.Relu(self.bn7(x))
        x = self.Linear3(x)
        x = self.Relu(self.bn8(x))
        x = self.Linear4(x)
        x = self.Relu(self.bn9(x))

        return x


if __name__ == '__main__':
    alex = AlexNet()
    x = torch.randn([16, 3, 227, 227])
    print(alex(x))