"""
    File name: trainers.py
    Description: Many personal child trainers based on its father class ---- base trainer
                All you need to write here is the basic logic of training during every one of
                epoch

    Author: Botian Lan
    Time: 2022/01/24
"""

import numpy as np
from baseline.base_trainer import BaseTrainer
from utils import *


class Cifar10Trainer(BaseTrainer):
    def __init__(self, model, epoch, data_loader, loss_function, optimizer, device):
        self.device = device
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.init_kwargs = {
            'model': model,
            'epoch': epoch,
            'data_loader': data_loader
        }
        super(Cifar10Trainer, self).__init__(**self.init_kwargs)

    def _epoch_train(self, epoch):
        print("Epoch{}:".format(epoch))
        for batch_index, dataset in enumerate(self.data_loader):
            datas, labels = dataset
            if self.device == 'cuda':
                datas, labels = data_to_gpu(datas, labels)

            self.optimizer.zero_grad()
            output = self.model(datas)
            loss_val = self.loss_function(output, labels)
            loss_val.backward()
            self.optimizer.step()

