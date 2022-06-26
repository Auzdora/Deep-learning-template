import numpy as np
from PIL import Image
import os
import torch
from utils.general_utils import *

def crop(image,dx):
    lists = []
    for i in range(image.shape[0]):
        for x in range(image.shape[1] // dx):
            for y in range(image.shape[2] // dx):
                print(image[i,  x*dx : (x+1)*dx, y*dx : (y+1)*dx].shape)
                lists.append(list(image[i,  y*dx : (y+1)*dx,  x*dx : (x+1)*dx])) #这里的list一共append了20x12x12=2880次所以返回的shape是(2880,48,48)

    return np.array(lists)


def preprocess_data(path, format):
    def createFileList(filename, format=format):
        fileList = []
        print(filename)
        for root, dirs, files in os.walk(filename, topdown=False):
            for name in files:
                if name.endswith(format):
                    fullName = os.path.join(root, name)
                    fileList.append(fullName)
        return fileList
    myFileList = createFileList(path)
    list = []
    for file in myFileList:
        print(file)
        img_file = Image.open(file)
        img_file = img_file.resize((1440,960))
        value = np.asarray(img_file.getdata(), dtype=np.int).reshape(960, 1440, -1)
        print(value.shape)
        value = crop_center(np.squeeze(value[...,1]), 960, 960)
        list.append(value)
        print(value.shape)
    list = np.asarray(list)
    return list


def preprocess_label(path, format):
    def createFileList(filename, format=format):
        fileList = []
        print(filename)
        for root, dirs, files in os.walk(filename, topdown=False):
            for name in files:
                if name.endswith(format):
                    fullName = os.path.join(root, name)
                    fileList.append(fullName)
        return fileList
    myFileList = createFileList(path)
    list = []
    for file in myFileList:
        print(file)
        img_file = Image.open(file)
        img_file = img_file.resize((1440,960))
        value = np.asarray(img_file.getdata(), dtype=np.int).reshape(960, 1440, -1)
        print(value.shape)
        value = crop_center(np.squeeze(value),960, 960)
        list.append(value)
        print(value.shape)
    list = np.asarray(list)
    return list


def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]


dx = 48
image_dataset = preprocess_data('database/data/image', '.png')
image_dataset = image_dataset.astype(dtype=np.uint8)
image_label = preprocess_label('database/data/label', '.png')

X_train = np.array(image_dataset)
Y_train = np.array(image_label)

X_train = X_train.astype('float32')/255
print(X_train.shape)
Y_train = Y_train.astype('float32')/255
img = Image.fromarray(X_train[40][48*11:48*12, 48*3:48*4])
img.show()
img = Image.fromarray(X_train[40])
img.show()
img = Image.fromarray(Y_train[40])
img.show()
label_y = Y_train[40][48*11:48*12, 48*3:48*4]
label_y[label_y>0.5] = 0.7
ss = torch.tensor(np.expand_dims(np.expand_dims(label_y,axis=0),axis=0))
dir_path = './database/result/'
save_image_tensor(ss, dir_path+'splist.png')
label_y[label_y>0.5] = 255
img = Image.fromarray(label_y)
img.show()
X_train = crop(X_train,dx)
Y_train = crop(Y_train,dx)

X_train = X_train[:,np.newaxis, ...]
Y_train = Y_train[:,np.newaxis, ...]
print('X_train shape: '+str(X_train.shape))
print('Y_train shape: '+str(Y_train.shape))

train_test_ratio = 0.8
len_of_data = X_train.shape[0]
X_trains = X_train[:int(train_test_ratio*len_of_data),...]
X_tests = X_train[int(train_test_ratio*len_of_data):,...]
Y_trains = Y_train[:int(train_test_ratio*len_of_data),...]
Y_tests = Y_train[int(train_test_ratio*len_of_data):,...]
print(X_trains.shape)
print(X_tests.shape)
print(Y_trains.shape)

