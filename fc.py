import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from scipy.misc import imresize
from torch.nn.parameter import Parameter
import torch.utils.data as data
from torchvision import transforms

from resnet_data.inceptionresnetv2.pytorch_load import inceptionresnetv2, InceptionResnetV2
from cloud_bm_v2 import ToTensor, Normalization, AmazonDataSet, read_data, train, Scale, RandomHorizontalFlip, RandomVerticalFlip, RandomSizedCrop
from resnet import get_resnet

class FC(nn.Module):
	def __init__(self):
		super(FC, self).__init__()
		self.fc1 = nn.Linear(50*50*3, 25*25*3)
		self.dropout1 = nn.Dropout(.3)
		self.fc2 = nn.Linear(25*25*3, 4)
	def forward(self, x):
		x = x.view(-1, 50*50*3)
		x = self.fc1(x)
		x = self.dropout1(x)
		x = self.fc2(x)
		return x

if __name__ == "__main__":
    print("Started")
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_weights", default=None, type=str)
    parser.add_argument("--img_dir", default="train/train-tif-v2/", type=str)
    args = parser.parse_args()
    batch_size = 64
    
    in_res = FC()
    #res = get_resnet([1,2,3], 13)
    #res.load_state_dict(torch.load("resnet.pth.tar"))
    print("Batch size: {}".format(batch_size))

    data_transform = transforms.Compose([
        Scale(50),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        #RandomSizedCrop(),
        ToTensor(),
        Normalization(),
    ])
    val_transform = transforms.Compose([
        Scale(50),
        ToTensor(),
        Normalization()])

    if args.load_weights:
        in_res.load_state_dict(torch.load(args.load_weights))
        print("Loaded weights from {}".format(args.load_weights))

    img_labels, features_gt, cloud_gt  = read_data("train.csv")
    val_img, val_features, val_cloud = read_data("validation.csv")

    cloud_data = AmazonDataSet(img_labels, cloud_gt, args.img_dir,4, transform=data_transform)
    validation_cloud_data = AmazonDataSet(val_img, val_cloud, args.img_dir,4, transform=val_transform)

    dataset_loader = data.DataLoader(cloud_data, batch_size=batch_size, shuffle=True, num_workers=16)
    validation_loader = data.DataLoader(validation_cloud_data, batch_size=batch_size, shuffle=True, num_workers=16)

    train(in_res, dataset_loader, validation_loader, batch_size, "CrossEntropy")



#end
