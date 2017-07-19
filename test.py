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


def read_data(filename):
	img = []
	feat = []
	cloud = []
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
			#Filler
			feature_one_hot = np.zeros(13)
			cloud_one_hot = np.zeros(1)
			feat.append(feature_one_hot)
			cloud.append(cloud_one_hot)

	return img, feat, cloud

########### Dataset #################################################
class AmazonDataSet(Dataset):

    def __init__(self, img_list, labels_list, root_dir, channels, transform=None):
        self.images = img_list
        self.labels = labels_list
        self.root_dir = root_dir
        self.channels = channels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        img_name = os.getcwd()+"/../train/train-tif-v2/" + self.images[idx] + ".tif"
        tif = TIFF.open(img_name, mode='r')
        image = tif.read_image()
        sample = {'image':image, 'labels': self.labels[idx]}
        #print(sample)
        if(self.transform):
            sample = self.transform(sample)
        sample['image'] = sample['image'].narrow(0,0,self.channels)
        return sample

################# End Dataset #######################################
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
    def __call__(self, sample):
        x = Image.fromarray(imresize(sample['image'], (299, 299)))
        return {'image': np.array(x), 'labels': sample['labels']}

test_transform = transforms.Compose([
    Scale(),
    ToTensor(),
    Normalization()])

def squeezenet():
    model = squeezenet1_1(pretrained=True, num_classes=1000)
    x = model.features[0].weight.data.numpy()
    s = x.shape
    l = []
    for i in s:
        l.append(i)
    l[1] += 1
    y = np.ones(tuple(l))
    for i in range(3):
        y[:, i, :] = x[:, i, :]
    y[:, 3, :] = (x[:, 0, :]+x[:, 1, :]+x[:, 2, :])/3.0
    model.features[0].weight.data = torch.from_numpy(y).float()
    model.features[0].in_channels = 4
    return model
############### End Custom Transforms ########################

def test_data(dataset_loader, filename):
	cloud_labels=['haze', 'clear', 'cloudy', 'partly_cloudy']
	feature_labels=['primary', 'agriculture', 'water', 'habitation', 'road', 'cultivation', 'slash_burn', 'conventional_mine', 'bare_ground', 'artisinal_mine', 'blooming', 'selective_logging', 'blow_down']

	########### Configure Model ############################
	squeezemodel = squeezenet()
	squeezemodel.load_state_dict(torch.load("model_squeezecloud.pth.tar"))
	resnet_model = get_resnet([0,1,2,3], 13)
	in_res.load_state_dict(torch.load("model_resnetppp.pth.tar"))
	if torch.cuda:
	    squeezemodel.cuda()
	    resnet_model.cuda()
	
	squeezemodel.eval()
	renset_model.eval()

	for batch in dataset_loader:
		inputs = batch['image']
		if torch.cuda:
			inputs = inputs.cuda()
		print(inputs)
		#RUN THROUGH SQUEEZENET MODEL FIRST
		outputs = squeezemodel(inputs)
		print(outputs)











test_file = os.getcwd()+ "/submission.csv"                                              #change to PATH_TO_FILE_FROM_CURRENT_DIRECTORY
test_img_labels, test_features_gt, test_cloud_gt  = read_data(test_file)                   #image filenames, feature and cloud ground truth arrays
test_cloud = AmazonDataSet(test_img_labels, test_cloud_gt, "/../test/test-tif-v2/", 4, transform=test_transform)
test_loader = DataLoader(test_cloud, batch_size=1, shuffle=False, num_workers=1)

test_data(test_loader, "output.csv")
