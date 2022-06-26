"""
    File name: trainers.py
    Description: Personal child trainers based on its father class ---- base trainer.
                All you need to write here is the basic logic of training during every one of
                epoch. Rewrite _epoch_train() method, for version 1.0, you need to write 2
                logic, one is train logic and another is test logic.

    Author: Botian Lan
"""

import numpy as np
import torch
from PIL import Image

from baseline.base_trainer import BaseTrainer
from utils.general_utils import *


def compute_score(predicts, label, n_class=2):

    assert (predicts.shape == label.shape)
    dice=[]
    PC=[]
    SE=[]
    Jaccard=[]
    SP=[]
    F1=[]
    acc = []
    overlap = ((predicts == 255) * (label == 255)).sum().float()  # TP
    b_overlap = ((predicts == 0) * (label == 0)).sum().float()  # TN
    FP = ((predicts == 255) * (label == 0)).sum().float()
    FN = ((predicts == 0) * (label == 255)).sum().float()
    if (overlap > 0):
        for i in range(predicts.shape[1]):
            acc.append((overlap+b_overlap)/(overlap+b_overlap+FP+FN))
            overlap = ((predicts[:, i, :, :] == 255) * (label[:, i, :, :] == 255)).sum().float()  # TP
            dice.append( 2 * overlap / ((predicts[:,i,:,:] == 255).sum() + (label[:,i,:,:] == 255).sum()).float())
            PC.append(overlap / (predicts[:,i,:,:] == 255).sum().float())
            SE.append(overlap / (label[:,i,:,:] == 255).sum().float())
            Jaccard.append(overlap / ((predicts[:,i,:,:] == 255).sum() + (label[:,i,:,:] == 255).sum() - overlap).float())
            SP.append(((predicts[:,i,:,:] == 0) * (label[:,i,:,:] == 0)).sum().float()/(label[:,i,:,:] == 0).sum().float())
            F1 .append(2*overlap / (label[:,i,:,:] == 255).sum().float()*overlap / (predicts[:,i,:,:] == 255).sum().float()/(overlap / (label[:,i,:,:] == 255).sum() + overlap / (predicts[:,i,:,:] == 255).sum().float() + 1e-6))
        return acc[0].item(), dice[0].item(), PC[0].item(), SE[0].item(), Jaccard[0].item(), SP[0].item()
    else:
        return 0, 0, 0, 0, 0, 0


class Cifar10Trainer(BaseTrainer):
    def __init__(self, model, epoch, data_loader, test_loader, loss_function, optimizer, lr, device, checkpoint_enable):
        self.test_data = test_loader
        self.device = device
        self.loss_function = loss_function
        self.lr = lr
        self.optimizer = optimizer(model.parameters(), lr=self.lr)
        self.init_kwargs = {
            'model': model,
            'epoch': epoch,
            'data_loader': data_loader,
            'optimizer': self.optimizer,
            'checkpoint_enable': checkpoint_enable,
            'device': device
        }
        super(Cifar10Trainer, self).__init__(**self.init_kwargs)

    def _epoch_train(self, epoch):

        total_train_loss = 0
        counter = 0
        tacc, tdice, tPC, tSE, tJaccard, tSP = 0, 0, 0, 0, 0, 0
        # train logic
        if epoch == 0:
            pass
        elif epoch % 12 == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.9
        for batch_index, dataset in enumerate(self.data_loader):
            # datas, labels = dataset
            datas = dataset[0]
            labels = dataset[1]
            # choose device
            if self.device == 'gpu':
                if torch.cuda.is_available():
                    datas, labels = data_to_gpu(datas, labels)
            elif self.device == 'cpu':
                pass

            output = self.model(datas)
            loss_val = self.loss_function(output, labels)
            # ACC, Dice, PC, SE, Jaccard, SP = compute_score(output, labels)

            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()

            total_train_loss += loss_val
            # tPC += PC
            # tSE += SE
            # tJaccard += Jaccard
            # tSP += SP
            # tacc += ACC
            # tdice += Dice
            counter += 1
            self.train_logger.info("Train {}: loss:{} ".format(counter,loss_val))#  ACC, Dice,  PC, SE, Jaccard, SP))
            self.console_logger.info("Train {}: loss:{} ".format(counter,loss_val))# ACC, Dice,  PC, SE, Jaccard, SP))
            # if counter % 100 == 0:
            #     self.train_logger.info("Train {}: loss:{}".format(counter, loss_val))
            #     self.console_logger.info("Train {}: loss:{}".format(counter, loss_val))

        total_test_loss, mean_test_loss, test_acc, test_dice, test_PC, test_SE, test_Jac, testSP = self._epoch_val(epoch)

        # self.train_logger.info('Epoch{}:--------loss:{}, test loss:{}, accuracy:{}'.format(epoch, total_train_loss,
        #                                             total_test_loss, total_accuracy.item()/self.test_data.length()))
        # self.console_logger.info('Epoch{}:--------loss:{}, test loss:{}, accuracy:{}'.format(epoch, total_train_loss,
        #                                             total_test_loss, total_accuracy.item() / self.test_data.length()))
        self.train_logger.info('Epoch{}:--------loss:{}, test loss:{} ACC:{}, DICE:{},PC:{} SE:{} Jac:{} SP:{}'.format(epoch, total_train_loss,
                                                                              mean_test_loss, test_acc, test_dice, test_PC, test_SE, test_Jac, testSP))
        self.console_logger.info('Epoch{}:--------loss:{}, test loss:{} ACC:{}, DICE:{},PC:{} SE:{} Jac:{} SP:{}'.format(epoch, total_train_loss,
                                                                              mean_test_loss, test_acc, test_dice, test_PC, test_SE, test_Jac, testSP))

    def _epoch_val(self, epoch):
        total_test_loss = 0
        tacc, tdice, tPC, tSE, tJaccard, tSP = 0, 0, 0, 0, 0, 0
        dir_path = './database/result/result_epoch{}'.format(epoch)
        mkdir(dir_path)
        with torch.no_grad():
            self.model.eval()
            cnt = 0
            img_buffer = []
            label_buffer = []
            for data in self.test_data:
                # datas, labels = data
                datas = data[0]
                labels = data[1]
                # choose device
                if self.device == 'gpu':
                    if torch.cuda.is_available():
                        datas, labels = data_to_gpu(datas, labels)
                elif self.device == 'cpu':
                    pass

                outputs = self.model(datas)

                loss = self.loss_function(outputs, labels)
                print('test loss:{} '.format(loss.item()))
                img_buffer.append(255 * np.array(outputs.cpu()))
                label_buffer.append(np.array(labels.cpu()))
                total_test_loss += loss.item()

                cnt += 1
                # accuracy = ((outputs.argmax(1) == labels).sum())
                # total_accuracy += accuracy

            t = 0

            img_buffer = np.array(img_buffer).squeeze(axis=1).squeeze(axis=1)
            for m in range(29):
                image = np.zeros((512, 512)) # 576, 576
                real = np.zeros((512, 512))
                for j in range(512 //64):
                    for i in range(512 // 64):
                        temp = img_buffer[t]
                        real[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64] = label_buffer[t]
                        image[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64] = temp
                        t = t + 1
                image = np.expand_dims(image,axis=0)
                image = np.expand_dims(image, axis=0)
                # real = 255 * real
                # real = real.astype(dtype=np.uint8)
                # img = Image.fromarray(real)
                # img.show()
                save_image_tensor(torch.tensor(image), dir_path + '/net_preeyes_{}.png'.format(m))
                img = Image.open('./database/result/result_epoch{}/net_preeyes_{}.png'.format(epoch,m)).convert('L')
                label = Image.open('./database/data/resizelabel/img{}.png'.format(71+m))
                img = np.array(img)
                label = np.array(label)
                label = torch.tensor(label)
                img = torch.tensor(img)
                acc, dice, pc, se, jac, sp = compute_score(torch.unsqueeze(torch.unsqueeze(img, dim=0), dim=0),
                                    torch.unsqueeze(torch.unsqueeze(label, dim=0), dim=0))
                tPC += pc
                tSE += se
                tJaccard += jac
                tSP += sp
                tacc += acc
                tdice += dice
                print(se)
            return total_test_loss, total_test_loss / float(cnt), tacc / float(29), tdice / float(29), tPC/float(29), tSE/float(29), tJaccard/float(29), tSP/float(29)


if __name__ =="__main__":
    list = []
    for data_file in sorted(os.listdir('E:/Melrose-Windows/Code/Python/Deeplearning_template_code/Deep_learning_template_code/database/data/resizeimg')):
        list.append(data_file)
    print(list)