"""
    File name: train.py
    Description: This file could help user train the network in a very straight way,
                all you have to do in here is point run button and wait.

    Author: Botian Lan
    Time: 2022/01/24
"""
import torch.cuda

from trainer import *
from logger_parse import *
from model.backbone import CIFAR_10_Model, Exp_net
from torch import nn, optim
from config_parse import _ConfigParser
from utils import loss, MyDataSet1, file_loader
import logging
from data_loader import data_loaders


def Launch_Engine(config):
    """
        Launch_Engine are combined with two code part, one is configuration part, other is
    'start to train' part.
        For configuration part, it does several things:
        1. logger_packer: pack 'log_config.json' file up, it'll call logger_parser function
    inside. logger_parser function will convert json to dict, at same time, by using logging.
    config.dictConfig() method, it'll initialize logging system.
        2. After initializing logging system, Launch_Engine will get logger by using logging.
    getLogger() method.
        3. Call class _ConfigParser's method to get config data
        4. If gpu device is available, move model to gpu.
        5. Pack useful information as a dict.
        6. If checkpoint enabled, begin to train model from the last iteration.
        7. If not, begin to train model from the beginning.
    :param config: an instantiation object
    :return: None
    """
    logger_packer('logger/log_config.json')

    # get logger
    console_logger = logging.getLogger('console_loggers')
    model_logger = logging.getLogger('model_file_loggers')

    model_logger.info('--------------------------Data loader information--------------------------')

    # get (dict) config information
    data_config = config.data_config
    model_config = config.model_config
    checkpoint_enable = config.checkpoint_enable

    # get data loader and test data loader
    Dataloader = getattr(data_loaders, data_config['data_loader']['train_dataLoader'])
    Test_Dataloader = getattr(data_loaders, data_config['data_loader']['test_dataLoader'])

    # load data set
    if config.my_dataset:
        train_set = MyDataSet1(data_config['train_db'], data_config['db_root'],
                               data_config['data_size'], data_config['label_size'], loader=file_loader(data_config))
        test_set = MyDataSet1(data_config['test_db'], data_config['db_root'],
                              data_config['data_size'], data_config['label_size'], loader=file_loader(data_config))

        data_loader = Dataloader(train_set, batch_size=data_config['train_batch_size'],
                                 shuffle=data_config['train_shuffle'])
        test_loader = Test_Dataloader(test_set, batch_size=data_config['test_batch_size'],
                                      shuffle=data_config['test_shuffle'])

    else:
        data_loader = Dataloader(data_config['train_data_path'], batch_size=data_config['train_batch_size'],
                                 shuffle=data_config['shuffle'])
        test_loader = Test_Dataloader(data_config['test_data_path'], batch_size=data_config['test_batch_size'],
                                      shuffle=data_config['shuffle'])

    # get other information
    epoch = model_config['epoch']
    loss_function = getattr(loss, model_config['loss_function'])
    optimizer = getattr(optim, model_config['optimizer'])
    learning_rate = model_config['lr']
    device = model_config['device']

    # get model
    my_model = getattr(Exp_net, model_config['model'])
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

    # Add another utils


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
    # Instantiation config parser and pass this parameter to Launch engine
    config = _ConfigParser('config.json')
    Launch_Engine(config)
