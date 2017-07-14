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
from cloud_parallel import ToTensor, Normalization, AmazonDataSet, read_data, train

f = InceptionResnetV2.forward
def forward(self, x):
	x = f(self, x)
	x = self.act(x)
	x = self.last(x)
	return x
InceptionResnetV2.forward = forward

parser = argparse.ArgumentParser()
parser.add_argument("--load_weights", default=None, type=str)
parser.add_argument("--img_dir", default="train/train-tif-v2/", type=str)
args = parser.parse_args()

in_res = inceptionresnetv2()
in_res.act = nn.ReLU(inplace=False)
in_res.last = nn.Linear(1001, 13)

class Scale(object):
    def __call__(self, sample):
        x = imresize(sample['image'], (299, 299))
        return {'image': x, 'labels': sample['labels']}
data_transform = transforms.Compose([
    Scale(),
    ToTensor(),
    Normalization(),
])

x = in_res.conv2d_1a.conv.weight.data.numpy()
s = x.shape;
print(s)
l = []
for i in s:
	l.append(i)
l[1] += 1
y = np.ones(tuple(l))
for i in range(3):
	y[:, i, :] = x[:, i, :]
y[:, 3, :] = (x[:, 0, :]+x[:, 1, :]+x[:, 2, :])/3.0
in_res.conv2d_1a.conv.weight.data = torch.from_numpy(y).float()
in_res.conv2d_1a.conv.in_channels = 4

if args.load_weights:
	print("Loaded weights from {}".format(args.load_weights))
	in_res.load_weights(args.load_weights)

img_labels, features_gt, _  = read_data("../train/train_v2.csv")
transformed_cloud_data = AmazonDataSet(img_labels, features_gt, args.img_dir,4, transform=data_transform)
dataset_loader = data.DataLoader(transformed_cloud_data, batch_size=32, shuffle=True, num_workers=16)

train(in_res, dataset_loader)
