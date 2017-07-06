#Nikhil Sardana
#7/5/17

import csv
import os
import numpy as np
from libtiff import TIFFfile, TIFFimage, TIFF
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def read_data(filename, cloud_labels, feature_labels):
	img = []
	feat = []
	cloud = []
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

    def __init__(self, img_list, labels_list, transform=None):
        self.transform=transform

    def __getitem__(self,idx):
        img_name = os.getcwd()+"/../train/train-tif-v2/" + img_list[idx] + ".tif"
        tif = TIFF.open('img_name', mode='r')
        image = tif.read_image()
        sample = {'image':image, 'labels': labels[idx]}
        #work in progress

def compute_statistics(image_list):
    s_d = np.asarray([1784.32824087, 1601.11714294, 1648.08226633, 1868.17758328])
    means = np.asarray([4988.75695801, 4270.74552917, 3074.87910461, 6398.84899376])
    m_ax = 0 #bad variable name
    for i in range(len(image_list)):
        img_path = os.getcwd() + "/../train/train-tif-v2/" + image_list[i] + ".tif"
        tif = TIFF.open(img_path, mode='r')
        ar = tif.read_image()
        temp_max = np.amax(ar)
        if(temp_max > m_ax):
            m_ax = temp_max
            print(m_ax)
    quit()
cloud = ['haze', 'clear', 'cloudy', 'partly_cloudy']
features = ['primary', 'agriculture', 'water', 'habitation', 'road', 'cultivation', 'slash_burn', 'conventional_mine', 'bare_ground', 'artisinal_mine', 'blooming', 'selective_logging', 'blow_down']
data_file = os.getcwd()+ "/../train/train_v2.csv" #change to PATH_TO_FILE_FROM_CURRENT_DIRECTORY
print(data_file)
img_labels, cloud_gt, features_gt = read_data(data_file, cloud, features) #image filenames, cloud and feature ground truth arrays
compute_statistics(img_labels)




#end file





