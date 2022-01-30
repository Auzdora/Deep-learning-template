"""
    File name: base_trainer.py
    Description: Base class for data loaders, you can create your own dataloader(class) based on this one.
                It could simplify your work.

    Author: Botian Lan
    Time: 2022/01/24
"""
from abc import abstractmethod, ABC
import logging


class BaseTrainer(ABC):
    def __init__(self, model, epoch, data_loader):
        self.model = model
        self.epoch = epoch
        self.data_loader = data_loader
        self.console_logger = logging.getLogger('console_loggers')
        self.train_logger = logging.getLogger('train_file_loggers')

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
            self.console_logger.info(' Epoch {} begin'.format(epoch))
            self.train_logger.info(' Epoch {} begin'.format(epoch))
            self._epoch_train(epoch)


if __name__ == '__main__':
    """
        Abstractmethod test code.
    """
    class Train(BaseTrainer):
        def __init__(self, model, epoch, data_loader):
            super(Train, self).__init__(model, epoch, data_loader)

    train = BaseTrainer(1, 2, 3)


