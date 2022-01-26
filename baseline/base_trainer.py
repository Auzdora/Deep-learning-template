"""
    File name: base_trainer.py
    Description: Base class for data loaders, you can create your own dataloader(class) based on this one.
                It could simplify your work.

    Author: Botian Lan
    Time: 2022/01/24
"""
from abc import abstractmethod, ABC


class BaseTrainer(ABC):
    def __init__(self, model, epoch, data_loader):
        self.model = model
        self.epoch = epoch
        self.data_loader = data_loader

    @abstractmethod
    def _epoch_train(self, epoch):
        """
         Train process for every epoch.
         Should be overridden by all subclasses.
        :param epoch: Specific epoch for one iteration.
        """
        raise NotImplementedError

    # TODO: ADD LOGGER
    def _train(self):
        for epoch in range(self.epoch):
            print('--------Epoch{} begin--------'.format(self.epoch))
            self._epoch_train(epoch)


if __name__ == '__main__':
    """
        Abstractmethod test code.
    """
    class Train(BaseTrainer):
        def __init__(self, model, epoch, data_loader):
            super(Train, self).__init__(model, epoch, data_loader)

    train = BaseTrainer(1,2,3)


