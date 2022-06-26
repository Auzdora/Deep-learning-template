"""
    File name: data_loaders.py
    Description: This file could combine BaseDataLoader to create your own loader.
                For instance:
                    You could create your own composer to preprocess data using DataLoader.Compose.
                    You could Load your own dataset.
                    ...

    Author: Botian Lan
"""
import torchvision
from base_data_loader import BaseDataLoader
from torchvision import transforms


class _DataLoader(BaseDataLoader):
    def __init__(self, root, batch_size, shuffle, num_workers=1):
        """
            Define your self data processing serials, also caller transformer.
        """
        transformer = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(std=(0.5, 0.5, 0.5), mean=(0.5, 0.5, 0.5))
             #transforms.RandomCrop()
            ]
        )
        self.dataset = torchvision.datasets.CIFAR10(root, train=True, transform=transformer, target_transform=None, download=True)
        super().__init__(self.dataset, batch_size, shuffle, num_workers)


class _Test_DataLoader(BaseDataLoader):
    def __init__(self, test_root, batch_size, shuffle, num_workers=1):
        """
            Define your self data processing serials, also caller transformer.
        """
        transformer = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(std=(0.5, 0.5, 0.5),mean=(0.5, 0.5, 0.5))
             #transforms.RandomCrop()
            ]
        )
        self.test_dataset = torchvision.datasets.CIFAR10(test_root, train=False, transform=transformer, target_transform=None, download=True)
        super().__init__(self.test_dataset, batch_size, shuffle, num_workers)


class _DataLoader2(BaseDataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers=1):
        """
            Define your self data processing serials, also caller transformer.
        """
        super().__init__(dataset, batch_size, shuffle, num_workers)


class _Test_DataLoader2(BaseDataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers=1):
        """
            Define your self data processing serials, also caller transformer.
        """
        super().__init__(dataset, batch_size, shuffle, num_workers)


if __name__ == '__main__':
    from logger.logger_parse import *
    root = 'D:/python/DL_Framework/database/train_data'
    test_root = 'D:/python/DL_Framework/database/test_data'
    logger_parser('D:/python/DL_Framework/logger/log_config.json')
    dataload = _DataLoader(root, 64, True, num_workers=1)
    print(len(dataload))
