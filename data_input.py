#Nikhil Sardana 7/5/17
import csv

def read_data(filename, cloud_labels, feature_labels):
	img = []
	feat = []
	cloud = []
	with open(filename, newline='') as csvfile:
		next(csvfile)
		datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		for row in datareader:
			second_split = row[0].split(',') #split on the comma between filename and first label
			img.append(second_split[0]) #image filename

			string_feat= [second_split[1]] + row[1:]

			cloud_one_hot = [0]*4			#one hot vectors
			feature_one_hot = [0]*13
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










cloud = ['haze', 'clear', 'cloudy', 'partly_cloudy']
features = ['primary', 'agriculture', 'water', 'habitation', 'road', 'cultivation', 'slash_burn', 'conventional_mine', 'bare_ground', 'artisinal_mine', 'blooming', 'selective_logging', 'blow_down']
data_file = "train_v2.csv" #change to PATH_TO_FILE_FROM_CURRENT_DIRECTORY
img_labels, cloud_gt, features_gt = read_data(data_file, cloud, features) #image filenames, cloud and feature ground truth arrays
