"""
    File name: train.py
    Description: This file could help user train the network in a very straight way,
                all you have to do in here is point run button and wait.

    Author: Botian Lan
"""
import numpy as np
import torch.cuda

from trainer import *
from logger_parse import *
from model.backbone import U_net_pack
from torch import nn, optim
from config_parse import _ConfigParser
from utils import loss, MyDataSet1, file_loader, MyDataset2
import logging
from data_loader import data_loaders
from PIL import Image


def Launch_Engine(config, X_trains, X_tests, Y_trains, Y_tests):
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
        # train_set = MyDataSet1(data_config['train_db'], data_config['db_root'],
        #                        data_config['data_size'], data_config['label_size'], loader=file_loader(data_config))
        # test_set = MyDataSet1(data_config['test_db'], data_config['db_root'],
        #                       data_config['data_size'], data_config['label_size'], loader=file_loader(data_config))
        train_set = MyDataset2(X_trains, Y_trains)
        test_set = MyDataset2(X_tests, Y_tests)

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
    my_model = getattr(U_net_pack, model_config['model'])
    model = my_model(n_channels=1, n_classes=1)


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
    def crop(image, dx):
        lists = []
        for i in range(image.shape[0]):
            for x in range(image.shape[1] // dx):
                for y in range(image.shape[2] // dx):
                    print(image[i, x * dx: (x + 1) * dx, y * dx: (y + 1) * dx].shape)
                    lists.append(list(image[i, y * dx: (y + 1) * dx,
                                      x * dx: (x + 1) * dx]))  # 这里的list一共append了20x12x12=2880次所以返回的shape是(2880,48,48)

        return np.array(lists)


    def preprocess_data(path, format):
        def createFileList(filename, format=format):
            fileList = []
            print(filename)
            for root, dirs, files in os.walk(filename, topdown=False):
                for name in files:

                    fullName = os.path.join(root, name)
                    fileList.append(fullName)
            return fileList

        myFileList = sorted(createFileList(path))
        list = []
        cnt=0
        for file in myFileList:
            print(file)
            # img_file = Image.open(file)
            # img_file = img_file.resize((1440, 960))
            # value = np.asarray(img_file.getdata(), dtype=np.int).reshape(960, 1440, -1)
            # print(value.shape)
            # value = crop_center(np.squeeze(value[..., 1]), 960, 960)
            # value = value.astype(dtype=np.uint8)
            # imgs = Image.fromarray(value)
            #
            # imgs = imgs.resize((512, 512))
            # imgs.save('./database/data/resizeimg/img{}.png'.format(cnt))
            # cnt += 1
            # val = np.asarray(imgs.getdata(), dtype=np.int).reshape((512,512,-1))
            # list.append(val)
            # print(val.shape)
            img_file = Image.open(file)
            value = np.asarray(img_file.getdata(), dtype=np.int).reshape(512, 512, -1)
            list.append(value)
            print(value.shape)
        list = np.asarray(list)
        return list


    def preprocess_label(path, format):
        def createFileList(filename, format=format):
            fileList = []
            print(filename)
            for root, dirs, files in os.walk(filename, topdown=False):
                for name in files:
                    fullName = os.path.join(root, name)
                    fileList.append(fullName)
            return fileList

        myFileList = sorted(createFileList(path))
        list = []
        cnt = 0
        for file in myFileList:
            print(file)
            # img_file = Image.open(file)
            # img_file = img_file.convert('L')
            # img_file = img_file.resize((1440, 960)) # 1440, 960
            # value = np.asarray(img_file.getdata(), dtype=np.int).reshape(960, 1440, -1) # 960, 1440
            # print(value.shape)
            # value = crop_center(np.squeeze(value), 960, 960) # 960, 960
            # value = value.astype(dtype=np.uint8)
            # imgs = Image.fromarray(value)
            # imgs = imgs.resize((512, 512))
            # imgs.save('database/data/resizelabel/img{}.png'.format(cnt))
            # cnt += 1
            # val = np.asarray(imgs.getdata(), dtype=np.int).reshape((512,512,-1))
            # list.append(val)
            img_file = Image.open(file)
            value = np.asarray(img_file.getdata(), dtype=np.int).reshape(512, 512, -1)
            list.append(value)
            print(value.shape)
        list = np.asarray(list)
        return list


    def crop_center(img, cropx, cropy):
        y, x = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty:starty + cropy, startx:startx + cropx]


    dx = 64
    image_dataset = preprocess_data('database/data/resizeimg', '.jpg')
    # image_dataset = image_dataset.astype(dtype=np.uint8)
    image_label = preprocess_label('database/data/resizelabel', '.png')

    X_train = np.array(image_dataset)
    Y_train = np.array(image_label)

    X_train = X_train.astype('float32') / 255
    print(X_train.shape)
    Y_train = Y_train.astype('float32') / 255
    X_train = np.squeeze(X_train)
    Y_train = np.squeeze(Y_train)
    X_train = crop(X_train, dx)
    Y_train = crop(Y_train, dx)

    # X_train = X_train[:, np.newaxis, ...]
    # Y_train = Y_train[:, np.newaxis, ...]
    print('X_train shape: ' + str(X_train.shape))
    print('Y_train shape: ' + str(Y_train.shape))

    X_trains = X_train[:6336, ...]
    X_tests = X_train[6336:, ...]
    Y_trains = Y_train[:6336, ...]
    Y_tests = Y_train[6336:, ...]
    image = np.zeros((512, 512))  # 576, 576
    t = 0
    for j in range(512 // 64):
        for i in range(512 // 64):
            temp = X_tests[t]
            image[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64] = temp
            t = t + 1
    image = 255 * image
    image = image.astype(dtype=np.uint8)
    images = Image.fromarray(image)
    images.show()
    config = _ConfigParser('config.json')
    Launch_Engine(config, X_trains, X_tests, Y_trains, Y_tests)
