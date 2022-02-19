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
from data_loader import data_loaders


def starter(config):
    logger_packer('logger/log_config.json')

    # get logger
    console_logger = logging.getLogger('console_loggers')
    model_logger = logging.getLogger('model_file_loggers')

    model_logger.info('--------------------------Data loader information--------------------------')

    # get (dict) config information
    data_congig = config.data_config
    model_config = config.model_config
    checkpoint_enable = config.checkpoint_enable

    # get data loader and test data loader
    Dataloader = getattr(data_loaders, data_congig['data_loader']['train_dataLoader'])
    Test_Dataloader = getattr(data_loaders, data_congig['data_loader']['test_dataLoader'])
    data_loader = Dataloader(data_congig['train_data_path'], batch_size=data_congig['batch_size'],
                             shuffle=data_congig['shuffle'])
    test_loader = Test_Dataloader(data_congig['test_data_path'], batch_size=data_congig['batch_size'],
                                  shuffle=data_congig['shuffle'])

    # get other information
    epoch = model_config['epoch']
    loss_function = getattr(loss, model_config['loss_function'])
    optimizer = getattr(optim, model_config['optimizer'])
    learning_rate = model_config['lr']
    device = model_config['device']

    # get model
    my_model = getattr(CIFAR_10_Model, model_config['model'])
    model = my_model()

    # convert to gpu model
    if torch.cuda.is_available() and device == 'gpu':
        model = model.cuda()

    init_kwargs = {
        'model': model,
        'epoch': epoch,
        'data_loader': data_loader,
        'test_loader': test_loader,
        'loss_function': loss_function,
        'optimizer': optimizer,
        'lr': learning_rate,
        'device': device,
        'checkpoint_enable': checkpoint_enable
    }

    # checkpoint logic
    if checkpoint_enable:
        console_logger.info('Checkpoint enabled successfully, continue to train from last time!')
    else:
        info = {
            'model_name': config.json_data['model_name'],
            'epoch': epoch,
            'loss_function': model_config['loss_function'],
            'optimizer': model_config['optimizer'],
            'learning rate': learning_rate,
            'device': device
        }
        # record
        info_shower(console_logger, **info)
        info_shower(model_logger, **info)

        console_logger.info('--------------------------Start to train--------------------------')

    # start to train
    train = Cifar10Trainer(**init_kwargs)
    train._train()


def info_shower(logger, **kwargs):
    """
        This function just for saving numbers of line of code.
    :param logger: specific logger
    :param kwargs: dict data structure
    """
    logger.info('--------------------------Model information--------------------------')
    for info in kwargs:
        logger.info('{}: {}'.format(info, kwargs[info]))


if __name__ == '__main__':
    config = _ConfigParser('config.json')
    starter(config)
