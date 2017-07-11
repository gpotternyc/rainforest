#Nikhil Sardana
# Began 7/5/17
import csv
import os
import numpy as np
from libtiff import TIFFfile, TIFFimage, TIFF
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as functional
from torch.autograd import Variable
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--disable-cuda", action="store_true", default=False)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--momentum", type=float, default=0.5)
parser.add_argument("--save_every", type=int, default=5)

args = parser.parse_args()
if args.cuda and not torch.cuda.is_available():
    print("no cuda... :(")
    args.cuda = False


def read_data(filename, cloud_labels, feature_labels):
	img, feat, cloud = [], [], []

	with open(filename, 'rb') as csvfile:
		next(csvfile)
		datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		for row in datareader:
			second_split = row[0].split(',') #split on the comma between filename and first label
			img.append(second_split[0]) #image filename

			string_feat= [second_split[1]] + row[1:]

			cloud_one_hot = np.array([np.zeros(4)])			#one hot vectors
			feature_one_hot = np.array([np.zeros(13)])
			#look for features by iterating over labels and doing string comparison
			for element in string_feat:
				for index,k in enumerate(feature_labels):
					if element==k:
						feature_one_hot[0][index] = 1
				for index,k in enumerate(cloud_labels):
					if element==k:
						cloud_one_hot[0][index] = 1

			feat.append(feature_one_hot)
			cloud.append(cloud_one_hot)

	return(img, feat, cloud)


class AmazonDataSet(Dataset):

    def __init__(self, img_list, labels_list, root_dir, transform=None):
        self.images = img_list
        self.labels = labels_list
        self.root_dir = root_dir
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

        return sample

############## Custom Transforms ####################################
class ToTensor(object):
     def __call__(self, sample):
        i, l = sample['image'], sample['labels']
        #swap color axis because numpy image H x W x C, torch image C x H x W
        i = i.transpose((2,0,1))
        l = l.transpose((1,0))
        i = i/65536.0
        return {'image': torch.from_numpy(i), 'labels': torch.from_numpy(l)}

class Normalization(object):
    def __call__(self, sample):
        actual_normalization=transforms.Normalize(mean=[0.076124,0.065167,0.05692,0.09764],std=[0.027227,0.024431,0.025148,0.028507])
        return {'image': actual_normalization(sample['image']), 'labels': sample['labels']}

data_transform = transforms.Compose([
    ToTensor(), 
    Normalization()])
############### End Custom Transforms ###########################

cloud = ['haze', 'clear', 'cloudy', 'partly_cloudy']
features = ['primary', 'agriculture', 'water', 'habitation', 'road', 'cultivation', 'slash_burn', 'conventional_mine', 'bare_ground', 'artisinal_mine', 'blooming', 'selective_logging', 'blow_down']
data_file = os.getcwd()+ "/../train/train_v2.csv" #change to PATH_TO_FILE_FROM_CURRENT_DIRECTORY

img_labels, features_gt, cloud_gt  = read_data(data_file, cloud, features) #image filenames, feature and cloud ground truth arrays

cloud_data  = AmazonDataSet(img_labels, cloud_gt, "/../train/train-tif-v2/")
transformed_cloud_data = AmazonDataSet(img_labels, cloud_gt, "/../train/train-tif-v2/", transform=data_transform)
dataset_loader = DataLoader(transformed_cloud_data, batch_size=args.batch_size, shuffle=True, num_workers=16)

model = models.squeezenet1_0(pretrained=true)
if args.cuda:
    model.cuda()
opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

model.train()
for epoch in range(args.epochs):
    for batch_num, (img, target) in enumerate(dataset_loader):
        if args.cuda:
            img = img.cuda()
            target = target.cuda()
        img = normalize(img)
        img = Variable(img); target = Variable(target)
        opt.zero_grad()
        out = model(img)
        loss = functional.binary_cross_entropy(out, target)
        loss.backward()
        opt.step()
        if batch_num % 50 == 0:
            print("Epoch: {}, batch: {}".format(epoch, batch_num))
    if (epoch+1) % args.save_every == 0:
        torch.save(model, epoch)