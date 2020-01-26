from torchvision import models, transforms, datasets
import torch
from PIL import Image
import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def setup_bbox():
	# Taken from https://github.com/daviswer/fewshotlocal/blob/master/Setup.ipynb

	annopath = 'D:/noam_/Cornell/CS7999/iNaturalist/train_2017_bboxes.json'
	with open(annopath) as f:
		allinfo = json.load(f)
	annolist = allinfo['annotations']
	imagelist = allinfo['images']

	imgdict = dict()	# image path to id number
	for d in imagelist:
		path = d['file_name'][17:]
		imgdict[path] = d['id']

	annodict = dict() # im name to list of box_ids
	boxdict = dict() # box_id to box coords
	catdict = dict() # dict of numerical category codes / labels to corresponding list of image ids
	for d in annolist:
		im = d['image_id']
		boxid = d['id']
		cat = d['category_id']
		
		# Add box_id to image entry
		if im in annodict:
			annodict[im].append(boxid)
		else:
			annodict[im] = [boxid]
			
		# Add mapping from box_id to box
		boxdict[im] = d['bbox']
		
		# Add image to category set
		if cat in catdict:
			catdict[cat].add(im)
		else:
			catdict[cat] = set([im])
    
	print("Built annotation dictionaries")
	return annodict, boxdict, catdict, imgdict

def extract_bbox(img, impath, annodict, boxdict, catdict, imgdict):
	# Takes in an image and returns the resulting extracted bounding box region
	# Snippets taken from https://github.com/daviswer/fewshotlocal/blob/master/helpful_files/training.py
	splitpath = impath.split('/')
	shortpath = ''
	for i in range(-3,0):	# Extract the shortened path name used as the dictionary key in imgdict
		shortpath += splitpath[i] + '/'
	shortpath = shortpath[:-1]

	if shortpath not in imgdict:
		return None

	# Calculate the minimum and maximum coordinates of the bounding box in the image
	ID = imgdict[shortpath]		
	box = boxdict[ID]
	xmin = box[0]
	xmax = box[2]+xmin
	ymin = box[1]
	ymax = box[3]+ymin
	xmin_int = int(xmin)
	xmax_int = int(xmax)+1
	ymin_int = int(ymin)
	ymax_int = int(ymax)+1

	# Crop the image to the bbox area
	img = np.array(img)
	img_box = img[ymin_int:ymax_int, xmin_int:xmax_int,:]

	# Uncomment to show bounded images
	# f = plt.figure()
	# f.add_subplot(1,2,1)
	# plt.imshow(img)
	# f.add_subplot(1,2,2)
	# plt.imshow(img_box)
	# plt.show(block=True)

	return img_box

def test_images(image_path, labels, model, annodict, boxdict, catdict, imgdict):
	'''
	Tests all images in the image_path with the provided model.
	Returns:
		results dictionary with all results (i.e. the confidence of all possible labels)
	'''
	df = pd.DataFrame(columns = ['Label index','Class','Image Name','Confidence'])
	results = {}
	for image in os.listdir(image_path):
		results[image] = {}

		# Load input image
		img = Image.open(image_path+image)	
		img = img.convert('RGB')

		if annodict is not None:
			img = extract_bbox(img, image_path + image, annodict, boxdict, catdict, imgdict)
		
		# Test on image
		out = test_one_image(img, model)
		val, index = torch.max(out,1)			# get the top 1 result

		to_sort = torch.argsort(out).numpy().flatten()
		out_sorted = torch.sort(out)[0]

		confidence = torch.nn.functional.softmax(out, dim=1)[0]*100
		conf_sorted = confidence[to_sort]

		# Next 3 lines from https://www.learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/
		# _, indices = torch.sort(out, descending=True)
		# percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
		# [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]

		df.loc[len(df)] = [index.item(),labels[str(index.item())][1], 
			labels[str(index.item())][0], confidence[index].item()]

		results[image]['labels'] = [labels[str(idx)][1] for idx in to_sort]
		results[image]['vals'] = [round(o.item(), 3) for o in out_sorted.data.numpy().flatten()]
		results[image]['confs'] = [c.item() for c in conf_sorted.data.numpy().flatten()]

	return df, results

def test_one_image(img, model):
	img_t = transform(img)				# shape ([3, 224, 224])
	batch_t = torch.unsqueeze(img_t, 0)	# add dimension in position 0
										# shape ([1, 3, 224, 224])

	model.eval()		# put model in eval mode

	# Test on image
	out = model(batch_t)				# shape ([1, 1000])
	return out

def get_most_common_labels(table, n):
	'''
	Gets the n most common labels for the given class
	'''	
	# Which classes were labelled?
	all_labels = table['Class']
	all_conf = table['Confidence']
	unique_labels = np.unique(all_labels)
	
	# Plot the chosen labels by their frequency
	label_counts = np.zeros(len(unique_labels))
	label_conf = np.zeros(len(unique_labels))
	for i, l in enumerate(unique_labels):
		label_counts[i] = np.sum(all_labels == l)
		conf = all_conf[all_labels==l]
		avg_conf = sum(conf)/len(conf)
		label_conf[i] = avg_conf

	# Sort the data and return the top n
	to_sort = np.argsort(label_counts)

	return unique_labels[to_sort][-n:], label_conf[to_sort][-n:], label_counts[to_sort][-n:]

def test_inat(root_path, savefile, model, imagenet_labels, bbox = False):
	# Set up bounding box annotations for iNaturalist images
	if bbox:
		annodict, boxdict, catdict, imgdict = setup_bbox()	
	else:
		annodict=None; boxdict=None; catdict=None; imgdict=None

	# Loop through each biological group
	for typename in os.listdir(root_path):
		# Create directories if they don't already exist
		try:
			os.mkdir(os.getcwd() + '/alexnet_birdvid_results/' + typename + '/')
		except:
			continue

		# Loop through each class in the group
		for classname in tqdm(os.listdir(root_path+typename+'/')):
			iNat_results = {}
			path = root_path+typename+'/'+classname+'/'

			# try:
			table, dic = test_images(path, imagenet_labels, model, annodict, boxdict, catdict, imgdict)

			labels, confs, counts = get_most_common_labels(table, 0)	# get all results
			
			to_sort = np.argsort(counts)
			# Dictionary version, all results for each picture
			for im in dic:
				iNat_results[im] = {}
				iNat_results[im]['labels'] = list(dic[im]['labels'])
				iNat_results[im]['vals'] = list(dic[im]['vals'])
				iNat_results[im]['confs'] = list(dic[im]['confs'])

			# except:
			# 	message = typename + ',' + classname
			# 	iNat_results['labs'] = None
			# 	iNat_results['confs'] = None
			# 	iNat_results['vals'] = None
			# 	pass

			with open(savefile+typename+'/'+classname+'.json', 'w') as outfile:
				json.dump(iNat_results, outfile)

#### Plotting functions - obsolete, see plot_result_distribution.py
def plot_all_labels(table, title=None, save=None, showplot=True):
	'''
	Creates a plot of the label distribution in the provided table
	'''
	
	# Which classes were labelled?
	all_labels = table['Class']
	all_conf = table['Confidence']
	unique_labels = np.unique(all_labels)
	
	# Plot the chosen labels by their frequency
	label_counts = {}
	label_conf = {}
	for l in unique_labels:
		label_counts[l] = np.sum(all_labels == l)

		conf = all_conf[all_labels==l]
		avg_conf = sum(conf)/len(conf)
		label_conf[l] = avg_conf

	# Plot
	keys = np.array(list(label_counts.keys()))
	vals = np.array(list(label_counts.values()))
	confs = np.array(list(label_conf.values()))
	assert(label_conf.keys() == label_counts.keys())	
	to_sort = np.argsort(vals)
	colors = [[x/100, 0, 0] for x in confs[to_sort]]

	plt.figure(figsize=[20, 8])
	plt.scatter(range(len(keys)), vals[to_sort], c=colors)
	plt.xticks(np.arange(len(keys)), keys[to_sort], rotation=90, fontsize=10)
	if title is not None:
		plt.title(title)
	plt.xlabel('Image Label', fontsize=14)
	plt.ylabel('Label count', fontsize=14)
	plt.tight_layout()

	# Add confidence numbers
	# for i, txt in enumerate(confs[to_sort]):
	# 	plt.annotate(str(int(txt))+'%', 
	# 				xy=(range(len(keys))[i], vals[to_sort][i]),
	# 				xytext=(range(len(keys))[i]-0.5, vals[to_sort][i]),
	# 				fontsize='x-small',
	# 				rotation=0)

	if showplot:
		plt.show()

	if save is not None:
    		plt.savefig(save)

	plt.close()
	return plt

def	plot_by_confidence(table):
	'''
	Splits the data into confidence levels and plots each one separately
	in subplots
	'''
	
	all_labels = table['Class']
	all_conf = table['Confidence']

	# Plot the results by confidence tier
	fig = plt.subplots(figsize=(10,4))
	unique_labels = np.unique(all_labels)
	for i in range(50, 100, 10):
		to_plot = np.argwhere(all_conf > i).flatten()

		# Extract which data points are in this interval
		labels = all_labels[to_plot]
		conf = all_conf[to_plot]
		
		# Group data points by label
		label_counts = {}
		label_conf = {}
		for l in np.unique(labels):

			label_counts[l] = np.sum(labels == l)
			c = conf[labels==l]

			if len(c) == 0:
				break
			avg_conf = sum(c)/len(c)
			label_conf[l] = avg_conf

		# Sort by number of labels to plot
		labels_grouped = np.array(list(label_counts.keys()))
		counts = np.array(list(label_counts.values()))
		confs_grouped = np.array(list(label_conf.values()))

		to_sort = np.argsort(counts)

		# Create subplots
		plt.tight_layout()
		plt.subplot(1, 5, int(i/10)+-4)
		plt.plot(range(len(labels_grouped)), counts[to_sort], 'ro')
		plt.xticks(np.arange(len(labels_grouped)), labels_grouped[to_sort], rotation=90, fontsize=8)
		message = 'Confidence > '+str(i)+'%'
		plt.title(message, fontsize='small')
		
	plt.suptitle('Quercus agrifola (oak tree)')
	plt.show()
	return plt

def plot_split_label_conf(table, title=None, showplot=False):
	'''
	Plots the label distribution from the table while stratifying confidence levels
	'''

	# Which classes were labelled?
	all_labels = table['Class']
	all_conf = table['Confidence']
	unique_labels = np.unique(all_labels)
	
	# Plot the chosen labels by their frequency
	label_counts = {}
	label_conf = {}
	for l in unique_labels:
		label_counts[l] = np.sum(all_labels == l)

		conf = all_conf[all_labels==l]
		label_conf[l] = list(conf)

	# Plot
	keys = np.array(list(label_counts.keys()))
	vals = np.array(list(label_counts.values()))
	confs = np.array(list(label_conf.values()))
	assert(label_conf.keys() == label_counts.keys())	
	to_sort = np.argsort(vals)

	plt.figure(figsize=[20, 8])

	i = 0
	for l in keys[to_sort][-10:]:
		cur_confs = np.array(label_conf[l])
		a = np.sum(cur_confs <= 20)
		b = np.sum(np.logical_and(cur_confs > 20, cur_confs <= 40))
		c = np.sum(np.logical_and(cur_confs > 40, cur_confs <= 60))
		d = np.sum(np.logical_and(cur_confs > 60, cur_confs <= 80))
		e = np.sum(cur_confs > 80)
		plt.bar(np.arange(i,i+5), [a,b,c,d,e])
		i += 5

	plt.xticks(np.arange(5*len(keys[-10:])), np.tile(np.arange(0,100,20), len(keys)), rotation=90, fontsize=8)
	plt.legend(keys[to_sort][-10:])
	# plt.xticks(np.arange(len(keys)), keys[to_sort], rotation=90, fontsize=10)
	if title is not None:
		plt.title(title)
	plt.xlabel('Image Label', fontsize=14)
	plt.ylabel('Label count', fontsize=14)
	plt.tight_layout()

	# Add confidence numbers
	# for i, txt in enumerate(confs[to_sort]):
	# 	plt.annotate(str(int(txt))+'%', 
	# 				xy=(range(len(keys))[i], vals[to_sort][i]),
	# 				xytext=(range(len(keys))[i]-0.5, vals[to_sort][i]),
	# 				fontsize='x-small',
	# 				rotation=0)

	if showplot:
		plt.show()

	plt.close()
	return plt

####

if __name__ == '__main__':
	# print(dir(models))
	image_path = 'D:/noam_/Cornell/CS7999/iNaturalist/train_val_images/'
	# image_path = 'D:/noam_/Cornell/CS7999/Macaulay Library Birds - Frames/'
	save_path = os.getcwd() + '/alexnet_birdvid_results/' 

	# Load pretrained Alexnet
	alexnet = models.alexnet(pretrained=True)

	# Load imagenet labels as dictionary
	# with open('D:/noam_/Cornell/CS7999/imagenet_class_index.json', 'r') as f:
	# 	imagenet_labels = [line.strip() for line in f.readlines()]
	with open('D:/noam_/Cornell/CS7999/imagenet_class_index.json', 'r') as f:
		imagenet_labels = json.load(f)

	# Transform for input images
	transform = transforms.Compose([
		transforms.Resize(256),			# images should be 256x256
		transforms.CenterCrop(224),		# crop about the center to 224x224
		transforms.ToTensor(),			# convert to Tensor
		transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
		)])
	
	# Test the image from iNaturalist
	test_inat(image_path, save_path, alexnet, imagenet_labels, bbox=False)

	