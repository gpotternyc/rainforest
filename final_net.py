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
from fc import FC


class Final(nn.Module):
    def __init__(self):
        super(Final, self).__init__()
        self.fc1 = nn.Linear(17, 50)
        self.fc2 = nn.Linear(50, 17)
        self.in_res = get_resnet([0,1,2,3], 13)
        #self.in_res.load_state_dict(torch.load(""))
        self.cloud = FC()
        #self.cloud.load_state_dict(torch.load("model_resnet.pth.tar"))

    def forward(self, x, y):
        x = self.in_res(x)
        y = self.cloud(y)
        print(x.size())
        z = torch.cat((x, y), 1)
        z = self.fc1(z)
        z = self.fc2(z)
        return z

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default="train/train-tif-v2/", type=str)
    args = parser.parse_args()
    
    batch_size = 17

    data_transform = transforms.Compose([
        Scale(),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        #RandomSizedCrop(),
        ToTensor(),
        Normalization(),
    ])
    val_transform = transforms.Compose([
        Scale(),
        ToTensor(),
        Normalization()])

    img_labels, features_gt, _  = read_data("train.csv")
    val_img, val_features, val_cloud = read_data("validation.csv")

    feature_data = AmazonDataSet(img_labels, features_gt, args.img_dir,4, transform=data_transform, return_scaled=True)
    validation_feature_data = AmazonDataSet(val_img, val_features, args.img_dir,4, transform=val_transform, return_scaled=True)

    dataset_loader = data.DataLoader(feature_data, batch_size=batch_size, shuffle=True, num_workers=16)
    validation_loader = data.DataLoader(validation_feature_data, batch_size=batch_size, shuffle=True, num_workers=16)


    m = Final()

    train(m, dataset_loader, validation_loader, batch_size, "MSE", input_dict=True)

#end
