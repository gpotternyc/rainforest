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
from squeezenet import squeezenet1_0
from scipy.misc import imresize
import sys
import shutil
import random
import time
from PIL import Image
use_crayon = True
if use_crayon:
	from pycrayon import CrayonClient


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

			cloud_one_hot = np.zeros(4)			#not one hot vectors
			feature_one_hot = np.zeros(13)
			#look for features by iterating over labels and doing string comparison
			for element in string_feat:
				for index,k in enumerate(feature_labels):
					if element==k:
						feature_one_hot[index] = 1
				for index,k in enumerate(cloud_labels):
					if element==k:
						cloud_one_hot[index] = 1

			feat.append(feature_one_hot)
			cloud.append(cloud_one_hot)

	return(img, feat, cloud)


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
    RandomSizedCrop(),
    ToTensor(), 
    Normalization()])
val_transform = transforms.Compose([
    Scale(),
    ToTensor(),
    Normalization()])
############### End Custom Transforms ########################
############### Validation ###################################
def validate(model, val_loader, batch_size):
    if torch.cuda:
        model.cuda()

    criterion = nn.MSELoss()
    model.eval()
    i=0
    running_loss = 0.0
    for batch in val_loader:
        i+=1
        inputs, targets = batch['image'], batch['labels']
        targets = torch.squeeze(targets)

        if torch.cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs = Variable(inputs); targets = Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.data[0]

    print("***Validation***")
    print(running_loss/(i*1.0))
    print("*End Validation*")

    model.train()
    return running_loss/(i*1.0)

############# End Validation ##################################

def save_checkpoint(state, is_best, filename="validation.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_resnet.pth.tar')

def precise(precision, best_prec, epoch, tot_batches, model, opt,i, is_train):
    is_best = precision > best_prec
    best_prec = min(best_prec, precision)
    if(is_train):
        if i%10 == 0:
            print(epoch+1, precision)
    else:
	    print("Writing Validation, tot_batches: {}".format(tot_batches))
	    print("Precision: {}, best precision: {}".format(precision, best_prec))
	    save_checkpoint(model.state_dict(), is_best, filename="validation-{}.pth.tar".format(tot_batches))

    return best_prec

LR = .001
steps = (2, 5, 10, 15)
def lr(opt, gamma, tot_batches, batches_per_epoch):
	st = 0
	for i in steps:
		if tot_batches / (batches_per_epoch+0.0) > i:
			st += 1
	new = LR * (gamma ** st)
	for p in opt.param_groups:
		p['lr'] = new
	 
def train(model, dataset_loader, val_loader, batch_size):
	x = 1000
	if use_crayon:
		c = CrayonClient(hostname="localhost")
		for _ in range(1000):
			try:
				d = c.create_experiment(str(x))
			except:
				x += 1
		print("Tensorboard: {}".format(x))
	if torch.cuda:
		model.cuda()
	opt = optim.SGD(model.parameters(), lr=LR, momentum=0.5)
	criterion = nn.MSELoss()
	model.train()
	best_prec = 2e15
	best_val = 2e15
	tot_batches = 0
	st = 0
	validate_every = int(.3 * len(dataset_loader))
	print("Validating every {} batches".format(validate_every))

	for epoch in range(50):
		running_loss = 0.0
		i=0
		for batch in dataset_loader:
			lr(opt, .3, tot_batches, len(dataset_loader))
			tot_batches += 1
			i+=1
			inputs, targets = batch['image'], batch['labels']
			targets = torch.squeeze(targets)
			
			if torch.cuda:
				inputs = inputs.cuda()
				targets = targets.cuda()
			inputs = Variable(inputs); targets = Variable(targets)
			opt.zero_grad()
			out = model(inputs)

			loss = criterion(out, targets)
			loss.backward()
			if use_crayon:
				d.add_scalar_value("loss", loss.data[0])
			opt.step()
			running_loss += loss.data[0]
            
			#Training Set Loss (Computationally Inexpensive)
			precision = running_loss/(i*1.0)
			best_prec = precise(precision, best_prec, epoch, tot_batches, model, opt, i, True)
			#if i % 2 == 1:
			if tot_batches % validate_every == validate_every-1:
			    print("Validation check")
			    precision=validate(model, val_loader, batch_size)
			    if use_crayon:
				    d.add_scalar_value("val loss", precision)
			    best_val = precise(precision, best_val, epoch, tot_batches, model, opt, i, False)
		
		#precision = validate(model, val_loader, batch_size)
		#best_val = precise(precision, best_val, epoch, tot_batches, model, opt, i, False)



if __name__ == "__main__":
    training_file = os.getcwd() + "/train.csv"
    img_labels, features_gt, cloud_gt = read_data(training_file)
    train_cloud = AmazonDataSet(img_labels, cloud_gt, "/../train/train-tif-v2/", 4, transform=data_transform)

    validation_file = os.getcwd()+ "/validation.csv"                                              #change to PATH_TO_FILE_FROM_CURRENT_DIRECTORY
    val_img_labels, val_features_gt, val_cloud_gt  = read_data(validation_file)                   #image filenames, feature and cloud ground truth arrays
    validation_cloud = AmazonDataSet(val_img_labels, val_cloud_gt, "/../train/train-tif-v2/", 4, transform=val_transform)

    dataset_loader = DataLoader(train_cloud, batch_size=32, shuffle=True, num_workers=16)
    print("Data Loaded")
    validation_loader = DataLoader(validation_cloud, batch_size=32, shuffle=True, num_workers=16)
    print("Validation Loaded")

    model = squeezenet1_0(pretrained=False, num_classes=4)
    train(model, dataset_loader, validation_loader, 32)


#end
