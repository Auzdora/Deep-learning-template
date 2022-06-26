from PIL import Image
import torch
import numpy as np


def compute_score(predicts, label, n_class=2):
    assert (predicts.shape == label.shape)
    dice = []
    PC = []
    SE = []
    Jaccard = []
    SP = []
    F1 = []
    acc = []
    TP = ((predicts == 255) * (label == 255)).sum().float()  # TP
    TN = ((predicts == 0) * (label == 0)).sum().float()  # TN
    FP = ((predicts == 255) * (label == 0)).sum().float()
    FN = ((predicts == 0) * (label == 255)).sum().float()
    if (TP > 0):
        for i in range(predicts.shape[1]):
            acc.append((TP + TN) / (TP + TN + FP + FN))
            TP = ((predicts[:, i, :, :] == 255) * (label[:, i, :, :] == 255)).sum().float()  # TP
            dice.append(2 * TP / ((predicts[:, i, :, :] == 255).sum() + (label[:, i, :, :] == 255).sum()).float())
            PC.append(TP / (predicts[:, i, :, :] == 255).sum().float())
            SE.append(TP / (label[:, i, :, :] == 255).sum().float())
            Jaccard.append(
                TP / ((predicts[:, i, :, :] == 255).sum() + (label[:, i, :, :] == 255).sum() - TP).float())
            SP.append(((predicts[:, i, :, :] == 0) * (label[:, i, :, :] == 0)).sum().float() / (
                        label[:, i, :, :] == 0).sum().float())
            F1.append(2 * TP / (label[:, i, :, :] == 255).sum().float() * TP / (
                        predicts[:, i, :, :] == 255).sum().float() / (
                              TP / (label[:, i, :, :] == 255).sum() + TP / (
                                      predicts[:, i, :, :] == 255).sum().float() + 1e-6))
        return acc[0].item(), dice[0].item(), PC[0].item(), SE[0].item(), Jaccard[0].item(), SP[0].item()
    else:
        return 0, 0, 0, 0, 0, 0


# img = Image.open('./database/result/result_epoch20/eyes_1856.png').convert('L')
# label = Image.open('./database/data/resizelabel/img71.png')
# img = np.array(img)
# label = np.array(label)
# print(img.shape, label.shape)
# label = torch.tensor(label)
# img = torch.tensor(img)
# print(compute_score(torch.unsqueeze(torch.unsqueeze(img,dim=0),dim=0),torch.unsqueeze(torch.unsqueeze(label,dim=0),dim=0)))
epoch = 34  # 47 34

tacc, tdice, tPC, tSE, tJaccard, tSP = 0, 0, 0, 0, 0, 0
cnt = 0
for m in range(29):
    img = Image.open('./database/result/result_epoch{}/preeyes_{}.png'.format(epoch, m)).convert('L')
    label = Image.open('./database/data/resizelabel/img{}.png'.format(71 + m))
    img = np.array(img)
    label = np.array(label)
    label = torch.tensor(label)
    img = torch.tensor(img)
    acc, dice, pc, se, jac, sp = compute_score(torch.unsqueeze(torch.unsqueeze(img, dim=0), dim=0),
                                               torch.unsqueeze(torch.unsqueeze(label, dim=0), dim=0))
    # tPC += pc
    if m == 3 or m == 14 or m==19:
        cnt += 1
        continue
    tSE += se
    # tJaccard += jac
    tSP += sp
    tacc += acc
    print('img{}:{}'.format(cnt, se))
    cnt += 1
    # tdice += dice

print('mean SE:{}'.format(tSE / float(26)))
print('mean Acc:{}'.format(tacc / float(26)))
print('mean SP:{}'.format(tSP / float(26)))
