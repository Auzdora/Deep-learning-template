"""
    File name: _Flatten.py
    Torch version: 1.2.0
    Description: Torch 1.2.0 didn't provide flatten operation in module nn, this file write this function
            based on pytorch, to improve code efficiency. To make it more flexible, I named it _Flatten in-
            -stead of Flatten, so that it can distinguish from future version.

    Author: Botian Lan
"""

import torch
from torch import nn


class Flatten(nn.Module):
    """
        _Flatten Class
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, inputs):
        output = torch.flatten(inputs, 1, -1)
        return output
