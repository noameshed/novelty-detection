# Comparing ImageNet to iNaturalist dataset to find any overlaps

import os
import json
import numpy as np
import string

def parse_line(line):
	# Returns common and scientific names of the species at that line

	line = line.lower()
	# Ignore any links to images
	if 'ref' in line:
		# Includes an image link
		idx = line.find('ref')
		line = line[:idx]

	splitline = line.split(' - ')
	for i in range(len(splitline)):
		splitline[i] = splitline[i].strip()	# remove trailing spaces
		splitline[i] = splitline[i].strip(string.punctuation)	# remove punctuation

	common_name = splitline[0]
	scientific_name = None
	if len(splitline) > 1:
		scientific_name = splitline[1].split(']]\'\', \'\'[[')

	return common_name, scientific_name
	

def plants_to_dict(f):
	# Converts the txt page from wikipedia to a dictionary
	name_dict = {}
	com_name = ''
	for line in f:
		# Main category
		if line[:1] == '*':	
			com_name, sci_name = parse_line(line)
			name_dict[com_name] = {}
			name_dict[com_name]['name'] = sci_name
			name_dict[com_name]['subclasses'] = {}

		# Sub-categories
		if line[:2] == ':*':
			sub_com_name, sub_sci_name = parse_line(line)
			name_dict[com_name]['subclasses'][sub_com_name] = sub_sci_name

	return name_dict


if __name__ == '__main__':
	# Parse the plant scientific names txt from wiki
	f = open('plant_scientific_wiki.txt', 'r')
	plant_dict = plants_to_dict(f)

	# Creat list of iNat classes
	iNat_path = 'D:/noam_/Cornell/CS7999/iNaturalist/train_val_images/'
	iNat_labels_sci = []
	for item in os.listdir(iNat_path):
		path = iNat_path + item
		for label in os.listdir(path):
			iNat_labels_sci.append(label.lower())

	# List of ImageNet classes
	f = open('D:/noam_/Cornell/CS7999/imagenet_class_index.json', 'r')
	imagenet_class_idx = json.load(f)
	
	imagenet_labels = []
	for label in imagenet_class_idx.values():
		imagenet_labels.append(label[1])

	# Scientific names for plants
	# for i in imagenet_labels:
	# 	if i in plant_dict:
	# 		print(i, plant_dict[i]['name'], '\n')