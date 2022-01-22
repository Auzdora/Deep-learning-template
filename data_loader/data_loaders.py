"""
    This file could combine BaseDataLoader to create your own loader.
    For instance:
        You could create your own composer to preprocess data using DataLoader.Compose.
        You could Load your own dataset.
        ...

    Author: Botian Lan
    Time: 2022/01/21
"""
import torchvision
from base_data_loader import BaseDataLoader
from torchvision import transforms

root = 'D:/python/DL_Framework/database/data'


class _DataLoader(BaseDataLoader):
    def __init__(self, root, batch_size, shuffle, num_workers=1):
        """
            Define your self data processing serials, also caller transformer.
        """
        transformer = transforms.Compose(
            [transforms.ToTensor(),
             #transforms.Normalize()
             #transforms.RandomCrop()
            ]
        )
        self.dataset = torchvision.datasets.CIFAR10(root, train=True, transform=transformer, target_transform=None, download=True)
        super().__init__(self.dataset,batch_size,shuffle,num_workers)


if __name__ == '__main__':
    dataload = _DataLoader(root,4,shuffle=True)
    for data in dataload:
        imgs, targets = data
        print(imgs.shape)
        print(targets)