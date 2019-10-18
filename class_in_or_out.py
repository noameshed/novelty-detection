import json
import os
import pandas as pd

def create_label_csv(labels_dict):

	df = pd.DataFrame(columns = ['index', 'id', 'name', 'in_inaturalist'])

	for i in labels_dict:
		imagenet_index = i
		imagenet_id = labels_dict[i][0]
		common_name = labels_dict[i][1]

		df.loc[len(df)] = [imagenet_index, imagenet_id, common_name, False]

	df.to_csv('in_out_classes.csv')

if __name__ == '__main__':

	# Load imagenet labels as dictionary
	f = open('D:/noam_/Cornell/CS7999/imagenet_class_index.json', 'r')
	imagenet_labels = json.load(f)

	create_label_csv(imagenet_labels)

	

		