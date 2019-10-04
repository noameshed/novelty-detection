from torchvision import models, transforms
import torch
from PIL import Image
import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def test_images(image_path, net):
	df = pd.DataFrame(columns = ['Label index','Class','Image Name','Confidence'])
	for image in os.listdir(image_path):
		# Load input image
		img = Image.open(image_path+image)	

		img_t = transform(img)				# shape ([3, 224, 224])
		batch_t = torch.unsqueeze(img_t, 0)	# add dimension in position 0
											# shape ([1, 3, 224, 224])

		net.eval()		# put model in eval mode

		# Test on image
		out = net(batch_t)				# shape ([1, 1000])

		val, index = torch.max(out,1)			# get the top result
		# print(val, index.item())

		confidence = torch.nn.functional.softmax(out, dim=1)[0]*100
		df.loc[len(df)] = [index.item(),labels[str(index.item())][1], 
			labels[str(index.item())][0], confidence[index].item()]

	return df


def plot_all_labels(path, model, title=None, showplot=False):
	try:
		table = test_images(path, model)
	except:
		print('Can not plot')
		pass


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

	plt.savefig('unlabeled_plots/'+typename+'/'+classname+'.png')
	plt.close()

def	plot_by_confidence(path, model):
	table = test_images(path, model)
	all_labels = table['Class']
	all_conf = table['Confidence']

	# Plot the results by confidence tier
	# fig = plt.figure(figsize=(10, 5))
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

def plot_split_label_conf(path, model, title=None, showplot=False):
	try:
		table = test_images(path, model)
	except:
		print('Can not plot')
		pass


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



if __name__ == '__main__':
	# print(dir(models))

	# Load pretrained AlexNet
	alexnet = models.alexnet(pretrained=True)
	# print(alexnet)

	# Load imagenet labels as dictionary
	f = open('D:/noam_/Cornell/CS7999/imagenet_class_index.json', 'r')
	labels = json.load(f)

	# Transform for input images
	transform = transforms.Compose([
		transforms.Resize(256),			# images should be 256x256
		transforms.CenterCrop(224),		# crop about the center to 224x224
		transforms.ToTensor(),			# convert to Tensor
		transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
		)])
	
	# path = 'D:/noam_/Cornell/CS7999/iNaturalist/train_val_images/Plantae/Rosa californica/'
	path = 'D:/noam_/Cornell/CS7999/iNaturalist/train_val_images/Reptilia/Terrapene carolina/'
	# path = 'D:/noam_/Cornell/CS7999/iNaturalist/train_val_images/Mollusca/Limacia cockerelli/'
	# path = 'D:/noam_/Cornell/CS7999/iNaturalist/train_val_images/Aves/Spheniscus demersus/'
	# path = 'D:/noam_/Cornell/CS7999/iNaturalist/train_val_images/Plantae/Woodwardia areolata/'
	# path = 'D:/noam_/Cornell/CS7999/iNaturalist/train_val_images/Plantae/Quercus agrifolia/'

	# for t in os.listdir('D:/noam_/Cornell/CS7999/iNaturalist/train_val_images/'):
	# 	typename = t
	# 	for c in tqdm(os.listdir('D:/noam_/Cornell/CS7999/iNaturalist/train_val_images/'+t)):
	# 		classname = c

	# 		path = 'D:/noam_/Cornell/CS7999/iNaturalist/train_val_images/' + typename + '/' + classname + '/'
			

	plot_split_label_conf(path, alexnet, title='Box turtle', showplot=True)