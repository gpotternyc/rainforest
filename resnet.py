import os
import argparse
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.utils.data as data

from resnet_data.inceptionresnetv2.pytorch_load import inceptionresnetv2, InceptionResnetV2
from cloud_bm_v2 import data_transform, AmazonDataSet, read_data, train

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
t = nn.Conv2d(4, 32, kernel_size=3, stride=2)
x = in_res.conv2d_1a.conv.weight.data.clone()
s = x.size();
l = []
for i in s:
	l.append(i)
l[1] += 1
y = torch.Tensor(*tuple(l))
for i in range(3):
	y[:, i, :] = x[:, i, :]
y[:, 3, :] = (x[:, 0, :]+x[:, 1, :]+x[:, 2, :])/3.0
t.weight.data = y
in_res.conv2d_1a.conv = t

if args.load_weights:
	print("Loaded weights from {}".format(args.load_weights))
	in_res.load_weights(args.load_weights)

img_labels, features_gt, _  = read_data(os.path.join(args.img_dir, "..", "train_v2.csv"))

features_ds = AmazonDataSet(img_labels, features_gt, args.img_dir)
transformed_cloud_data = AmazonDataSet(img_labels, features_gt, args.img_dir, transform=data_transform)
dataset_loader = data.DataLoader(transformed_cloud_data, batch_size=32, shuffle=True, num_workers=16)

train(in_res, dataset_loader)
