"""
    File name: general_utils.py
    Description: Provides some general tools for programming, aim at improving code extendability

    Author: Botian Lan
    Time: 2022/01/25
"""
import torch


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


if __name__ == '__main__':
    data = torch.tensor([1, 2, 3])
    label = torch.tensor([0])
    data, label = data_to_gpu(data, label)
    print(data)
    print(label)
