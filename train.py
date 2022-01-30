"""
    File name: train.py
    Description: This file could help user train the network in a very straight way,
                all you have to do in here is point run button and wait.

    Author: Botian Lan
    Time: 2022/01/24
"""

from trainer import *
from data_loader.data_loaders import _DataLoader, _Test_DataLoader
from logger_parse import *
from model.backbone import CIFAR_10_Model
from torch import nn, optim
from config_parse import _ConfigParser
from utils import loss
import logging


def starter(config):
    logger_packer('D:/python/DL_Framework/logger/log_config.json')

    data_congig = config.data_config
    model_config = config.model_config

    data_loader = _DataLoader(data_congig['train_data_path'], batch_size=data_congig['batch_size'],
                              shuffle=data_congig['shuffle'])
    test_loader = _Test_DataLoader(data_congig['test_data_path'], batch_size=data_congig['batch_size'],
                                   shuffle=data_congig['shuffle'])

    my_model = getattr(CIFAR_10_Model, model_config['model'])
    if torch.cuda.is_available():
        my_model = my_model().cuda()

    epoch = model_config['epoch']
    loss_function = getattr(loss, model_config['loss_function'])
    optimizer = getattr(optim, model_config['optimizer'])
    learning_rate = model_config['lr']
    device = model_config['device']

    init_kwargs = {
        'model': my_model,
        'epoch': epoch,
        'data_loader': data_loader,
        'test_loader': test_loader,
        'loss_function': loss_function,
        'optimizer': optimizer,
        'lr': learning_rate,
        'device': device
    }

    # get logger
    console_logger = logging.getLogger('console_logger')
    model_logger = logging.getLogger('model_file_loggers')

    # record
    console_logger.info('model_name: {}'.format(config.json_data['model_name']))
    console_logger.info('epoch: {}'.format(epoch))
    console_logger.info('loss_function: {}'.format(model_config['loss_function']))
    console_logger.info('optimizer: {}'.format(model_config['optimizer']))
    console_logger.info('learning rate: {}'.format(learning_rate))
    console_logger.info('device: {}'.format(device))

    model_logger.info('model_name: {}'.format(config.json_data['model_name']))
    model_logger.info('epoch: {}'.format(epoch))
    model_logger.info('loss_function: {}'.format(model_config['loss_function']))
    model_logger.info('optimizer: {}'.format(model_config['optimizer']))
    model_logger.info('learning rate: {}'.format(learning_rate))
    model_logger.info('device: {}'.format(device))

    train = Cifar10Trainer(**init_kwargs)
    train._train()


if __name__ == '__main__':

    config = _ConfigParser('config.json')
    starter(config)