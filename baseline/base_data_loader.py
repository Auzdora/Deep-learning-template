"""
    Base class for data loaders, you can create your own dataloader(class) based on this one.
    It could simplify your work.

    Author: Botian Lan
    Time: 2022/01/21
"""

import numpy as np
from torch.utils.data import DataLoader


class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        super().__init__(dataset,batch_size,shuffle,num_workers=num_workers)

    # TODO Add random split method
    def spliter(self):
        pass
