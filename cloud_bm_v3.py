#Nikhil Sardana
# Began 7/5/17
import csv
import os
import numpy as np
from libtiff import TIFFfile, TIFFimage, TIFF
import torch
from torch.tensor import _TensorBase
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms, utils
import torch.nn.functional as functional
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from squeezenet import squeezenet1_1, SqueezeNet
from scipy.misc import imresize
import sys
import shutil
import random
import time
from PIL import Image
#use_crayon = True
use_crayon = False
if use_crayon:
        from pycrayon import CrayonClient
from cloud_bm_v2 import *

class_weights = torch.Tensor([3.752132, 0.35593191, 4.84418382, 1.39367856])
if torch.cuda:
        class_weights = class_weights.cuda()

class_weights = None

data_transform = transforms.Compose([
    Scale(),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    #RandomSizedCrop(),
    ToTensor(), 
    Normalization()])
val_transform = transforms.Compose([
    Scale(),
    ToTensor(),
    Normalization()])
############### End Custom Transforms ########################
############### Validation ###################################
############# End Validation ##################################

def save_checkpoint(state, is_best, filename="validation.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_squeeze_feature.pth.tar')

def precise(precision, best_prec, epoch, tot_batches, model, opt,i, is_train):
    is_best = precision > best_prec
    best_prec = min(best_prec, precision)
    if(is_train):
        if i%10 == 0:
            print(epoch+1, precision)
    else:
            print("Writing Validation, tot_batches: {}".format(tot_batches))
            print("Precision: {}, best precision: {}".format(precision, best_prec))
            save_checkpoint(model.state_dict(), is_best, filename="squeeze_feature_validation-{}.pth.tar".format(tot_batches))

    return best_prec

LR = .0020
steps = (5, 15, 30, 60, 100, 150)
def lr(opt, gamma, tot_batches, batches_per_epoch):
        st = 0
        for i in steps:
                if tot_batches / (batches_per_epoch+0.0) > i:
                        st += 1
        new = LR * (gamma ** st)
        for p in opt.param_groups:
                p['lr'] = new
         

from cloud_bm_v2 import train, validate

if __name__ == "__main__":
    training_file = os.getcwd() + "/train.csv"
    img_labels, features_gt, cloud_gt = read_data(training_file)
    train_cloud = AmazonDataSet(img_labels, features_gt, "/../train/train-jpg/", 3, transform=data_transform)

    o = SqueezeNet.forward
    def forward(self, x):
        x = o(self, x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.last(x)
        x = self.sigmoid(x)
        return x
    SqueezeNet.forward = forward
    
    batch_size = 64

    validation_file = os.getcwd()+ "/validation.csv"                                              #change to PATH_TO_FILE_FROM_CURRENT_DIRECTORY
    val_img_labels, val_features_gt, val_cloud_gt  = read_data(validation_file)                   #image filenames, feature and cloud ground truth arrays
    validation_cloud = AmazonDataSet(val_img_labels, val_features_gt, "/../train/train-jpg/", 3, transform=val_transform)

    dataset_loader = DataLoader(train_cloud, batch_size=batch_size, shuffle=True, num_workers=16)
    print("Data Loaded")
    validation_loader = DataLoader(validation_cloud, batch_size=batch_size, shuffle=True, num_workers=16)
    print("Validation Loaded")

    model = squeezenet1_1(pretrained=True, num_classes=1000)
    model.last = nn.Linear(1000, 13)
    model.dropout = nn.Dropout(.4)
    model.sigmoid = nn.Sigmoid()
    model.relu = nn.ReLU()
    train(model, dataset_loader, validation_loader, batch_size)


#end
