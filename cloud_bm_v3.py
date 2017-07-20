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

class_weights = torch.Tensor([3.752132, 0.35593191, 4.84418382, 1.39367856])
if torch.cuda:
        class_weights = class_weights.cuda()

class_weights = None

def read_data(filename, cloud_labels=['haze', 'clear', 'cloudy', 'partly_cloudy'],\
        feature_labels=['primary', 'agriculture', 'water', 'habitation', 'road', 'cultivation', 'slash_burn', 'conventional_mine', 'bare_ground', 'artisinal_mine', 'blooming', 'selective_logging', 'blow_down']):
        img, feat, cloud = [], [], []

        if sys.version_info[0] == 2:
                x = 'rb'
        else:
                x = 'r'
        with open(filename, x) as csvfile:
                next(csvfile)
                datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                for row in datareader:
                        second_split = row[0].split(',') #split on the comma between filename and first label
                        img.append(second_split[0]) #image filename

                        string_feat= [second_split[1]] + row[1:]

                        cloud_one_hot = np.zeros(1)                     #not one hot vectors
                        #cloud_one_hot = np.zeros(4)                    #not one hot vectors
                        feature_one_hot = np.zeros(13)
                        #look for features by iterating over labels and doing string comparison
                        for element in string_feat:
                                for index,k in enumerate(feature_labels):
                                        if element==k:
                                                feature_one_hot[index] = 1
                                for index,k in enumerate(cloud_labels):
                                        if element==k:
                                                cloud_one_hot[0] = index
                                                #cloud_one_hot[index] = 1

                        feat.append(feature_one_hot)
                        cloud.append(cloud_one_hot)

        return(img, feat, cloud)


class AmazonDataSet(Dataset):

    def __init__(self, img_list, labels_list, root_dir, channels, transform=None, return_scaled=False):
        self.images = img_list
        self.labels = labels_list
        self.root_dir = root_dir
        self.channels = channels
        self.transform = transform
        self.return_scaled = return_scaled
        if return_scaled:
            self.transform2 = transforms.Compose([])
            self.transform2.transforms = self.transform.transforms[:]
            self.transform2.transforms[0] = Scale(50)

    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        img_name = os.getcwd()+"/../train/train-jpg/" + self.images[idx] + ".jpg"
        image = np.array(Image.open(img_name))
        sample = {'image':image, 'labels': self.labels[idx]}
        p = sample
        #print(sample)
        if(self.transform):
            sample = self.transform(sample)
        sample['image'] = sample['image'].narrow(0,0,self.channels)
        if self.return_scaled:
            s = self.transform2(p)
            sample['image_scaled'] = s['image'].narrow(0,0,self.channels)
        return sample

############## Custom Transforms ####################################
class ToTensor(object):
     def __call__(self, sample):
        i, l = sample['image'], sample['labels']
        #swap color axis because numpy image H x W x C, torch image C x H x W
        i = i.transpose((2,0,1))
        i = i/65536.0
        return {'image': _TensorBase.float(torch.from_numpy(i)), 'labels': _TensorBase.float(torch.from_numpy(l))}

class Normalization(object):
    def __call__(self, sample):
        actual_normalization=transforms.Normalize(mean=[0.076124,0.065167,0.05692,0.09764],std=[0.027227,0.024431,0.025148,0.028507])
        return {'image': actual_normalization(sample['image']), 'labels': sample['labels']}

class Scale(object):
    def __init__(self, size=299):
        self.size = size
    def __call__(self, sample):
        x = Image.fromarray(imresize(sample['image'], (self.size, self.size)))
        return {'image': np.array(x), 'labels': sample['labels']}

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        x = np.array(sample['image'])
        y = sample['labels']
        if random.random() <= .5:
            x = x[::-1, :, :]
        return {'image': x, 'labels': y}

class RandomSizedCrop(object):
    def __init__(self):
        self.t = transforms.RandomSizedCrop(299)
    def __call__(self, sample):
        x = Image.fromarray(sample['image'])
        y = sample['labels']
        return {'image': np.array(self.t(x)), 'labels': y}

class RandomVerticalFlip(object):
    def __call__(self, sample):
        x = np.array(sample['image'])
        y = sample['labels']
        if random.random() <= .5:
            x = x[:, ::-1, :]
        return {'image': x, 'labels': y}

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
