import torch
from torch.nn import Linear, Sigmoid, CrossEntropyLoss
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Prepare data
male_heights = np.random.normal(171, 6, 500)
female_heights = np.random.normal(158, 5, 500)

male_weights = np.random.normal(70, 10, 500)
female_weights = np.random.normal(57, 8, 500)

male_bfrs = np.random.normal(16, 2, 500)
female_bfrs = np.random.normal(22, 2, 500)

male_labels = [1] * 500
female_labels = [0] * 500

height = np.concatenate((male_heights, female_heights))
weight = np.concatenate((male_weights, female_weights))
bfrs = np.concatenate((male_bfrs, female_bfrs))
labels = np.concatenate((male_labels, female_labels))


def prepro(input, set):
    return (input - np.mean(set)) / (np.max(set) - np.min(set))


# Preprocess and shuffle
train_set = np.array([prepro(height, height), prepro(weight, weight), prepro(bfrs, bfrs), labels]).T


class Mymodel(torch.nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.layer1 = Linear(in_features=3, out_features=3)
        self.layer2 = Linear(in_features=3, out_features=2)
        self.layer3 = Linear(in_features=2, out_features=1)

        self.sig = Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.sig(x)
        x = self.layer2(x)
        x = self.sig(x)
        x = self.layer3(x)
        return x


class Mydata(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_data = np.expand_dims(self.data[:, :-1], axis=-1)
        label = self.data[:, -1]
        return torch.tensor(input_data[index].T, dtype=torch.float), label[index]


class MyLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        super(MyLoader, self).__init__(self.dataset, batch_size, shuffle)


class LossMSE(torch.nn.Module):
    def __init__(self, batch_size):
        self.batch = batch_size
        super(LossMSE, self).__init__()

    def forward(self, pred, target):
        # b = (pred.squeeze(-1) - target.unsqueeze(-1))
        # a = (pred.squeeze(-1) - target) ** 2
        _sum = 0
        for index, tensor in enumerate(pred):
            a = target
            ins = tensor - target[index].unsqueeze(0)
            _sum += (((tensor.squeeze(-1) - target[index]) ** 2).mean())
        # loss = (((pred.squeeze(-1) - target.unsqueeze(-1)) ** 2).sum())/self.batch
        return _sum / self.batch


class CE(torch.nn.Module):
    def __init__(self, batch_size):
        self.batch = batch_size
        super(CE, self).__init__()

    def forward(self, pred, target):
        _sum = 0
        for index, tensor in enumerate(pred):
            a = target[index]
            s = torch.log(tensor).squeeze(1)
            asa = torch.mul(target[index], torch.log(tensor).squeeze(1))
            _sum += -torch.sum(torch.mul(target[index], torch.log(tensor).squeeze(1)))
        # loss = (((pred.squeeze(-1) - target.unsqueeze(-1)) ** 2).sum())/self.batch
        return _sum / self.batch


if __name__ == "__main__":

    label = torch.tensor([[0, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.float)
    loss_fn = CrossEntropyLoss()
    loss_fns = LossMSE(2)

    a = torch.tensor([[[1], [2], [1]], [[1], [1], [1]]], dtype=torch.float)
    c = torch.tensor([[[2,2,1], [2, 1, 0], [-1, 1, 1], [-2, 1, 2]]], dtype=torch.float, requires_grad=True)
    s = torch.tensor([[0], [1], [1], [-1]], dtype=torch.float, requires_grad=True)
    d = torch.matmul(c, a)
    l = torch.add(d, s)
    soft = torch.nn.Softmax(dim=1)
    o = soft(l)
    loss = loss_fns(o, label)
    l.retain_grad()
    loss.retain_grad()
    d.retain_grad()
    o.retain_grad()
    loss.backward()
    print(loss)
    print(o.grad)
    print(c.grad)
    # net = Mymodel()
    # mydata = Mydata(train_set)
    # dataloader = MyLoader(mydata, 2, True)
    # loss_fn = LossMSE(2)
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    #
    # for epoch in range(3000):
    #     total_train_loss = 0
    #     counter = 0
    #     for data, label in dataloader:
    #         output = net(data)
    #         loss_val = loss_fn(output, label)
    #         optimizer.zero_grad()
    #         loss_val.backward()
    #         optimizer.step()
    #         total_train_loss += loss_val
    #         counter += 1
    #
    #     print("epoch{}: loss:{}".format(epoch, total_train_loss/counter))