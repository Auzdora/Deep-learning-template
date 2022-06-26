"""
    File name: base_data_loader
    Description: Base class for data loaders, you can create your own dataloader(class) based on this one.
                It could simplify your work.

    Author: Botian Lan
"""

import numpy as np
from torch.utils.data import DataLoader
import logging


class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers
        }

        self.log_record()

        super().__init__(**self.init_kwargs)

    # TODO Add random split method
    def spliter(self):
        pass

    def length(self):
        return len(self.dataset)

    def log_record(self):
        # get loggers
        console_loggers = logging.getLogger('console_loggers')
        model_info_loggers = logging.getLogger('model_file_loggers')

        # data loader info
        console_loggers.info('Dataset length: {}'.format(len(self.dataset)))
        console_loggers.info('Batch size: {}'.format(self.init_kwargs['batch_size']))
        console_loggers.info('Shuffle: {}'.format(self.init_kwargs['shuffle']))
        console_loggers.info('Num workers: {}'.format(self.init_kwargs['num_workers']))

        model_info_loggers.info('Dataset length: {}'.format(len(self.dataset)))
        model_info_loggers.info('Batch size: {}'.format(self.init_kwargs['batch_size']))
        model_info_loggers.info('Shuffle: {}'.format(self.init_kwargs['shuffle']))
        model_info_loggers.info('Num workers: {}'.format(self.init_kwargs['num_workers']))
