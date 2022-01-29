from torch import nn


def loss_fn(output, labels):
    loss = nn.CrossEntropyLoss()
    return loss(output, labels)

