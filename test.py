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
from resnet import get_resnet
from scipy.misc import imresize
import sys
import shutil
import random
import time
from cloud_bm_v2 import AmazonDataSet
from PIL import Image

from cloud_bm_v2 import *

test_transform = transforms.Compose([
    Scale(),
    ToTensor(),
    Normalization()])
############# End Custom Transforms ##################################

############## SQUEEZENET IMPLEMENTATION ##############################
def squeezenet():

    o = SqueezeNet.forward
    def forward(self, x):
        x = o(self, x)
        x = self.dropout(x)
        x = self.last(x)
        x = self.sigmoid(x)
        return x
    SqueezeNet.forward = forward
    
    model = squeezenet1_1(pretrained=True, num_classes=1000)
    model.last = nn.Linear(1000,13)
    model.dropout = nn.Dropout(0.4)
    model.sigmoid = nn.Sigmoid()
    model.relu = nn.ReLU()

    model.load_state_dict(torch.load("model_resnet.pth.tar"))
    return model
############### End Squeezenet Implementation ########################
from fc import FC

def test_data(dataset_loader, filename):
    cloud_labels=['haze', 'clear', 'cloudy', 'partly_cloudy']
    feature_labels=['primary', 'agriculture', 'water', 'habitation', 'road', 'cultivation', 'slash_burn', 'conventional_mine', 'bare_ground', 'artisinal_mine', 'blooming', 'selective_logging', 'blow_down']

    ########### Configure Model ############################
    resnet_model = squeezenet()
    #resnet_model = get_resnet([0,1,2,3], 13)
    #resnet_model.load_state_dict(torch.load("model_resnetppp.pth.tar"))
    #cloud_model = FC()
    #cloud_model.load_state_dict(torch.load("model_resnet.pth.tar"))
    if torch.cuda:
        #squeezemodel.cuda()
        resnet_model.cuda()
        #cloud_model.cuda()    

    #squeezemodel.eval()
    resnet_model.eval()
    #cloud_model.eval()
    
    avg_f2 = 0
    num = 0
    for batch in dataset_loader:
        if num%100==0:
            print("!!!!" + str(num))
        true_positive, false_positive, false_negative = 0.0, 0.0, 0.0
        precision, recall = 0.0, 0.0
        f_2 = 0.0

        features = [0]*13
    
        inputs = batch['image']
        if torch.cuda:
            inputs = inputs.cuda()
        inputs = Variable(inputs)
        outputs = resnet_model(inputs)
        outputs = outputs.cpu().data.numpy()
        for k in range(13):
            if(outputs[0][k]>0.5):
                features[k]=1
        print(outputs[0])
        ground_truth = batch['labels']
        for j in range(13):
            if(features[j]==1 and ground_truth[0][j]==1):
                true_positive = true_positive + 1
            if(features[j]==1 and ground_truth[0][j]==0):
                false_positive = false_positive + 1
            if(features[j]==0 and ground_truth[0][j]==1):
                false_negative = false_negative + 1
        print(true_positive, false_positive, false_negative)
        if(true_positive!=0):
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            f_2 = 1.25 * ((precision * recall) / (precision + recall))
            avg_f2 = avg_f2 + f_2
            num = num + 1
    print(avg_f2/num)


test_file = os.getcwd()+ "/validation.csv"                                              #change to PATH_TO_FILE_FROM_CURRENT_DIRECTORY
test_img_labels, test_features_gt, test_cloud_gt  = read_data(test_file)                   #image filenames, feature and cloud ground truth arrays
test_features = AmazonDataSet(test_img_labels, test_features_gt, "/../train/train-jpg/", 4, transform=test_transform, return_scaled=True)
test_loader = DataLoader(test_features, batch_size=1, shuffle=False, num_workers=1)

test_data(test_loader, "output.csv")
