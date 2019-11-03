from torchvision import models, transforms, datasets
import torch
from PIL import Image
import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def test_images(image_path, labels, model):
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

		# Test on image
		out = test_one_image(img, model)
		
		val, index = torch.max(out,1)			# get the top 1 result

		to_sort = torch.argsort(out).numpy().flatten()
		out_sorted = torch.sort(out)[0]

		confidence = torch.nn.functional.softmax(out, dim=1)[0]*100
		conf_sorted = confidence[to_sort]

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

def plot_all_labels(table, title=None, save=None, showplot=False):
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

	
def test_inat(test_path, model, imagenet_labels):
	iNat_results = {}
	# Loop through each biological group
	for typename in os.listdir(test_path):
		if typename != 'Aves':
			continue
		# iNat_results[typename] = {}
		iNat_results = {}
		counter = 0

		# Loop through each creature in the group
		for classname in tqdm(os.listdir(test_path+typename+'/')):
			iNat_results[classname] = {}
			path = test_path+typename+'/'+classname+'/'

			try:
				table, dic = test_images(path, imagenet_labels, model)

				labels, confs, counts = get_most_common_labels(table, 0)	# get all results
				
				to_sort = np.argsort(counts)
				# Dictionary version, all results for each picture
				for im in dic:
					iNat_results[classname][im] = {}
					iNat_results[classname][im]['labels'] = list(dic[im]['labels'])
					iNat_results[classname][im]['vals'] = list(dic[im]['vals'])
					iNat_results[classname][im]['confs'] = list(dic[im]['confs'])

					# Table version, top result for each picture
					# iNat_results[typename][classname]['labels'] = labels[to_sort].tolist()
					# iNat_results[typename][classname]['confs'] = confs[to_sort].tolist()
					# iNat_results[typename][classname]['counts'] = counts[to_sort].tolist()
					# counter += 1

			except:
				# print('Can not plot', typename, classname)
				iNat_results[classname]['labels'] = None
				iNat_results[classname]['confs'] = None
				iNat_results[classname]['counts'] = None
				pass
		
			fname = 'alexnet_inat_results/inat_results_all_' + typename + '.json'
			with open(fname, 'w') as outfile:
				json.dump(iNat_results, outfile)

	# with open('inat_results_all.json', 'w') as outfile:
	# 	json.dump(iNat_results, outfile)
	
def load_json(filename):
	df = pd.DataFrame(columns=['Kingdom', 'Class','Common Name','Relation to Imagnet'])
	with open(filename) as json_file:
		data = json.load(json_file)
		for kingdom in data:
			for a in data[kingdom]:
				df.loc[len(df)] = [kingdom, a, None, None]

	df.to_csv('in_out_class.csv')


if __name__ == '__main__':
	# print(dir(models))
	test_path = 'D:/noam_/Cornell/CS7999/iNaturalist/train_val_images/'

	# Load pretrained Alexnet
	alexnet = models.alexnet(pretrained=True)
	# print(alexnet)

	# Load imagenet labels as dictionary
	f = open('D:/noam_/Cornell/CS7999/imagenet_class_index.json', 'r')
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
	

	# path = test_path + '/Plantae/Rosa californica/'
	# path = test_path + '/Reptilia/Terrapene carolina/'
	# path = test_path + '/Mollusca/Limacia cockerelli/'
	# path = test_path + '/Aves/Spheniscus demersus/'
	# path = test_path + '/Plantae/Woodwardia areolata/'
	# path = test_path + '/Plantae/Quercus agrifolia/'

	load_json('inat_results.json')

	test_inat(test_path, alexnet, imagenet_labels)

	