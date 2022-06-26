"""
    File name: config_writer.py
    Description: This file is useless for training, it is just a pre-work, write dict type of data to json file.
                In the future, user could just tune the params in config.json directly.

    Author: Botian Lan
"""
from utils_json import *

model_config = {
    'model_name': 'LeNet',
    'author': 'Botian Lan',
    'version': '1.0',

    'data': {
        'data_split': True,
        'split_data': {
            'train_data_path': './database/train_data',
            'test_data_path': './database/test_data',
            'data_loader': {
                'train_dataLoader': '_DataLoader',
                'test_dataLoader': '_Test_DataLoader'
            },
            'batch_size': 64,
            'shuffle': True
        },
        'original_data': {
            'batch_size': 64,
            'shuffle': True
        }
    },

    'model_params': {
        'model': 'LeNet',
        'epoch': 100,
        'data_loader': 'data_loader',
        'test_loader': 'test_loader',
        'loss_function': "loss_fn",
        'optimizer': "Adam",
        'lr': 0.001,
        'device': 'cuda'
    }
}

if __name__ == '__main__':
    data = read_json('config.json')
    print(data['data']['data_split'])