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


def get_resnet(device_ids, num_outs, sigmoid=True, dropout=True):
    f = InceptionResnetV2.forward
    def forward(self, x):
        x = f(self, x)
        x = self.act(x)
        if dropout:
            x = self.dropout1(x)
        x = self.last(x)
        if dropout:
            x = self.dropout2(x)
        if sigmoid:
            x = self.sigmoid(x)
        return x
    InceptionResnetV2.forward = forward

    in_res = inceptionresnetv2()
    in_res.act = nn.ReLU(inplace=False)
    in_res.last = nn.Linear(1001, num_outs)
    in_res.dropout1 = nn.Dropout(.3)
    in_res.dropout2 = nn.Dropout(.2)
    in_res.sigmoid = nn.Sigmoid()

    parallelize = ['conv2d_1a', 'conv2d_2a', 'conv2d_2b', 'conv2d_3b', 'conv2d_4a',
                    'mixed_5b', 'repeat', 'mixed_6a', 'repeat_1', 'mixed_7a',
                    'repeat_2', 'block8', 'conv2d_7b']
    m = []
    for x in parallelize:
        m.append(getattr(in_res, x))
        setattr(in_res, x, nn.DataParallel(m[-1], device_ids=device_ids))
    return in_res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_weights", default=None, type=str)
    parser.add_argument("--img_dir", default="train/train-jpg/", type=str)
    args = parser.parse_args()
    
    in_res = get_resnet([0,1,2,3], 13)
    batch_size = 54

    data_transform = transforms.Compose([
        Scale(),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomSizedCrop(),
        ToTensor(),
        Normalization(),
    ])
    val_transform = transforms.Compose([
        Scale(),
        ToTensor(),
        Normalization()])

    if args.load_weights:
        in_res.load_state_dict(torch.load(args.load_weights))
        print("Loaded weights from {}".format(args.load_weights))

    img_labels, features_gt, _  = read_data("train.csv")
    val_img, val_features, val_cloud = read_data("validation.csv")

    feature_data = AmazonDataSet(img_labels, features_gt, args.img_dir,4, transform=data_transform)
    validation_feature_data = AmazonDataSet(val_img, val_features, args.img_dir,4, transform=val_transform)

    dataset_loader = data.DataLoader(feature_data, batch_size=batch_size, shuffle=True, num_workers=16)
    validation_loader = data.DataLoader(validation_feature_data, batch_size=batch_size, shuffle=True, num_workers=16)

    train(in_res, dataset_loader, validation_loader, batch_size)



#end
