"""
    File name: base_trainer.py
    Description: Base class for training, you can create your own train (class) based on this one.
                It could simplify your work.

    Author: Botian Lan
"""
import os
from abc import abstractmethod, ABC
import logging
import torch
import pathlib


class BaseTrainer(ABC):
    def __init__(self, model, epoch, data_loader, optimizer, checkpoint_enable, device):
        self.checkpoint_enable = checkpoint_enable
        self.data_loader = data_loader
        self.console_logger = logging.getLogger('console_loggers')
        self.train_logger = logging.getLogger('train_file_loggers')
        self.model = model
        self.epoch = epoch
        self.optimizer = optimizer
        self.device = device

        if checkpoint_enable:
            self.last_epoch, self.checkpoint = self.load_model()
            self.model.load_state_dict(self.checkpoint['model'])
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])

    @abstractmethod
    def _epoch_train(self, epoch):
        """
         Train process for every epoch.
         Should be overridden by all subclasses.
        :param epoch: Specific epoch for one iteration.
        """
        raise NotImplementedError

    @abstractmethod
    def _epoch_val(self, epoch):
        """
         Train process for every epoch.
         Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def _train(self):
        """
         Epoch-wise train logic.
        :return:
        """
        for epoch in range(self.epoch):
            # checkpoint enabled
            if self.checkpoint_enable:
                # jump over last epoch
                if epoch <= self.last_epoch:
                    continue
                else:
                    self.console_logger.info(' Epoch {} begin'.format(epoch))
                    self.train_logger.info(' Epoch {} begin'.format(epoch))
                    self._epoch_train(epoch)
                    self.model.train()

                    # save model for every epoch
                    self.save_model(self.model, epoch)
            else:
                self.console_logger.info(' Epoch {} begin'.format(epoch))
                self.train_logger.info(' Epoch {} begin'.format(epoch))
                self._epoch_train(epoch)
                self.model.train()

                # save model for every epoch
                self.save_model(self.model, epoch)

    def save_model(self, model, epoch):
        """
         Save model's parameters of every epoch into pkl file.
        :param model: train model
        :param epoch: current epoch number
        :return:
        """
        path = 'model/saved_model/'
        model_path = 'model/saved_model/model_state_dict_{}.pkl'.format(epoch)
        model_saved_path = pathlib.Path(path)
        if model_saved_path.is_dir():
            model_info = self.checkpoint_generator(model, epoch)
            torch.save(model_info, 'model/saved_model/model_state_dict_{}.pkl'.format(epoch))
            self.console_logger.info('---------------- Model has been saved to "{}",epoch {} ----------------'
                                     .format(model_path, epoch))
            self.train_logger.info('---------------- Model has been saved to "{}",epoch {} ----------------'
                                   .format(model_path, epoch))

        else:
            raise FileNotFoundError("Model saved directory: '{}' not found!".format(path))

    def checkpoint_generator(self, model, epoch):
        checkpoint = {
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
            'model': model.state_dict()
        }

        return checkpoint

    def load_model(self):
        """
         Load saved model dict to
        :param epoch: choose a specific epoch number to load
        :return:
        """

        path = './model/saved_model/'
        # find the last model-saved file
        epoch_num = []
        for model_file in os.listdir(path):
            name, ext = os.path.splitext(model_file)
            epoch_num.append(int(name[-1]))
        last_epoch = max(epoch_num)

        # load checkpoint
        model_path = 'model/saved_model/model_state_dict_{}.pkl'.format(last_epoch)
        model_load_path = pathlib.Path(model_path)
        if model_load_path.is_file():
            if self.device == 'gpu':
                checkpoint = torch.load(model_load_path)
            elif self.device == 'cpu':
                checkpoint = torch.load(model_load_path, map_location='cpu')

            self.console_logger.info('---------------- Model parameters of epoch {} has been loaded ----------------'
                                     .format(last_epoch))
        else:
            raise FileNotFoundError("Model state dict: '{}' not found!".format(path))
        return last_epoch, checkpoint


if __name__ == '__main__':
    """
        Abstractmethod test code.
    """
    path = '../model/saved_model/'
    for model_file in os.listdir(path):
        name, ext = os.path.splitext(model_file)
        print(int(name[-1]))
    num = [1, 2]
    print(max(num))
