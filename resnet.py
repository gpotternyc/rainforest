import os
import argparse
import torch
import torch.utils.data as data

from resnet_data.inceptionresnetv2.pytorch_load import inceptionresnetv2
from cloud_bm_v2 import data_transform, AmazonDataSet, read_data

parser = argparse.ArgumentParser()
parser.add_argument("--load_weights", default=None, type=str)
parser.add_argument("--img_dir", default="train/train-tif-v2/", type=str)
args = parser.parse_args()

in_res = inceptionresnetv2()
x = in_res.conv2d_1a.weights 
in_res.conv2d_1a.weights = torch.Tensor(
                in_res.conv2d_1a.in_channels+1, in_res.conv2d_1a.out_channels // in_res.conv2d_1a.groups,
				*in_res.conv2d_1a.kernel_size)
for i in range(3):
	in_res.conv2d_1a.weights[i, :, :] = x[i]
in_res.conv2d_1a.weights[3, :, :] = (x[0]+x[1]+x[2])/3.0

if args.load_weights:
	print("Loaded weights from {}".format(args.load_weights))
	in_res.load_weights(args.load_weights)

img_labels, features_gt, _  = read_data(os.path.join(args.img_dir, "..", "train_v2.csv"))

features_ds = AmazonDataSet(img_labels, features_gt, args.img_dir)
transformed_cloud_data = AmazonDataSet(img_labels, features_gt, args.img_dir, transform=data_transform)
dataset_loader = data.DataLoader(transformed_cloud_data, batch_size=32, shuffle=True, num_workers=16)

