"""
    File name: general_utils.py
    Description: Provides some general tools for programming, aim at improving code extendability

    Author: Botian Lan
"""
import os
import torch
from torchvision import utils as vutils


# TODO: Maybe this line of code could be optimized. Q: How to return *args
def data_to_gpu(*args):
    """
     Only for data&label combination
    :param args:
    :return: buffer[0] is data, buffer[1] is label
    """
    buffer = []
    for val in args:
        if type(val) != 'tensor':
            val = torch.as_tensor(val)
        val = val.to(device='cuda')
        buffer.append(val)
    return buffer[0], buffer[1]


def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
        Save img for computer vision project. You could call it at
    epoch eval. By default, the result will restore into './database/
    result/'
    :param input_tensor:
    :param filename:
    :return:
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    vutils.save_image(input_tensor, filename)


def mkdir(path):
    """
        Create a folder.
    :param path: the folder path you wanna create.
    """
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


if __name__ == '__main__':
    pass
