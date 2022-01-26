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
from utils.general_utils import *


class Cifar10Trainer(BaseTrainer):
    def __init__(self, model, epoch, data_loader, test_loader, loss_function, optimizer, lr, device):
        self.test_data = test_loader
        self.device = device
        self.loss_function = loss_function
        self.lr = lr
        self.optimizer = optimizer(model.parameters(),lr=self.lr)
        self.init_kwargs = {
            'model': model,
            'epoch': epoch,
            'data_loader': data_loader
        }
        super(Cifar10Trainer, self).__init__(**self.init_kwargs)

    def _epoch_train(self, epoch):
        total_train_loss = 0
        counter = 0
        for batch_index, dataset in enumerate(self.data_loader):
            datas, labels = dataset
            if self.device == 'cuda':
                datas, labels = data_to_gpu(datas, labels)
            output = self.model(datas)
            loss_val = self.loss_function(output, labels)
            total_train_loss += loss_val
            counter += 1
            if counter % 100 == 0:
                print("Train {}: loss:{}".format(counter,loss_val))
            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()

        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data in self.test_data:
                imgs, labels = data
                if self.device == 'cuda':
                    imgs, labels = data_to_gpu(imgs, labels)
                outputs = self.model(imgs)
                loss = self.loss_function(outputs, labels)
                total_test_loss += loss.item()
                accuracy = ((outputs.argmax(1) == labels).sum())
                total_accuracy += accuracy
        print('Epoch{}:--------loss:{}, test loss:{}, accuracy:{}'.format(epoch, total_train_loss, total_test_loss, total_accuracy.item()/self.test_data.length()))




