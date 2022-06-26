import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import scipy.io as sio


def file_loader(config):
    extension = config['data_ext']
    if extension == '.mat':
        return MatLoader(config)
    else:
        return OtherLoader(config)


def image_loader(path):
    return Image.open(path).convert('L')
    # return Image.open(path).convert('L')


class MatLoader:
    def __init__(self, config):
        norm_input = config['norm_input']
        norm_label = config['norm_label']
        self.norm_flag = config['norm_flag']
        self.min_max_list = [norm_input, norm_label]

    def array_norm(self, data, index):
        if self.norm_flag is None:
            return data

        min_max = self.min_max_list[index]

        if min_max is None:
            return data

        if None in min_max:
            if min_max[0] != min_max[1]:
                return data
        elif min_max[0] >= min_max[1]:
            return data

        if (min_max[0] is None) and (min_max[1] is None):
            _range = np.max(data) - np.min(data)
            return (data - np.min(data)) / _range

        _range = min_max[1] - min_max[0]
        return (data - min_max[0]) / _range

    def __call__(self, path, index, **kwargs):
        ret = sio.loadmat(path)
        for item in ret.values():
            if isinstance(item, np.ndarray):
                return Image.fromarray(self.array_norm(item, index))


class OtherLoader:
    def __init__(self, config):
        norm_input = config['norm_input']
        norm_label = config['norm_label']
        self.norm_flag = config['norm_flag']
        self.min_max_list = [norm_input, norm_label]

    def array_norm(self, data, index):
        if self.norm_flag is None:
            return data

        min_max = self.min_max_list[index]

        if min_max is None:
            return data

        if None in min_max:
            if min_max[0] != min_max[1]:
                return data
        elif min_max[0] >= min_max[1]:
            return data

        if (min_max[0] is None) and (min_max[1] is None):
            _range = np.max(data) - np.min(data)
            return (data - np.min(data)) / _range

        _range = min_max[1] - min_max[0]
        return (data - min_max[0]) / _range

    def __call__(self, path, index, **kwargs):
        ret = image_loader(path)
        return self.array_norm(np.asarray(ret, dtype=np.float32), index)


class MyDataSet1(Dataset):
    def __init__(self, img_file, data_root, data_size, label_size, loader):
        imgs = []
        with open(img_file, 'r') as f:
            for line in f:
                temp = line.strip().split(' ')
                imgs.append((temp[0], temp[1]))

        self.data_root = data_root
        self.imgs = imgs
        self.data_size = data_size
        self.label_size = label_size
        self.loader = loader

    def __getitem__(self, index):
        fn1, labelfn = self.imgs[index]
        img = self.loader(os.path.join(self.data_root, fn1), 0)
        label = self.loader(os.path.join(self.data_root, labelfn), 1)
        label = label
        img = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.data_size[1], self.data_size[2])),
            transforms.ToTensor()])(img)
        label = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.label_size[1], self.label_size[2])),
            transforms.ToTensor()])(label)

        fn1 = fn1.split('\\')[1:]
        fn = fn1[0] + ',' + fn1[1]

        return img, label, fn

    def __len__(self):
        return len(self.imgs)


class MyDataset2(Dataset):
    def __init__(self, input, label):
        self.input = input
        self.label = label

    def __getitem__(self, index):
        input_img = self.input[index]
        input_label = self.label[index]
        img = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()])(input_img)
        label = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()])(input_label)
        return img, label

    def __len__(self):
        return len(self.input)

