"""
    File name: train.py
    Description: This file could help user train the network in a very straight way,
                all you have to do in here is point run button and wait.

    Author: Botian Lan
    Time: 2022/01/24
"""

from trainer import *
from baseline import *
from data_loader.data_loaders import _DataLoader, _Test_DataLoader
from model.backbone.CIFAR_10_Model import *
from torch import nn, optim


def starter(**kwargs):
    train = Cifar10Trainer(**kwargs)
    train._train()


if __name__ == '__main__':
    batch_size = 64
    shuffle = True
    store_path = 'D:/python/DL_Framework/database/data'
    test_root = 'D:/python/DL_Framework/database/test_data'
    data_loader = _DataLoader(store_path, batch_size=batch_size, shuffle=shuffle)
    test_loader = _Test_DataLoader(test_root, batch_size=batch_size, shuffle=shuffle)

    model = CIFAR10()
    if torch.cuda.is_available():
        model = model.cuda()
    init_kwargs = {
        'model': model,
        'epoch': 100,
        'data_loader': data_loader,
        'test_loader': test_loader,
        'loss_function': nn.CrossEntropyLoss(),
        'optimizer': optim.Adam,
        'lr': 0.001,
        'device': 'cuda'
    }
    starter(**init_kwargs)
