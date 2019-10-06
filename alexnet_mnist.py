# Testing results of alexnet on MNIST data
from torchvision import models, transforms, datasets
import torch
from PIL import Image
import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def test_on_mnist(model, labels):
	# Test model on MNIST
	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST('/files/', train=True, download=True,
                             	transform=transforms.Compose([
								transforms.Resize(256),
								transforms.CenterCrop(224),
                               	transforms.ToTensor(),
                               	transforms.Normalize(
                                 (0.1307,), (0.3081,))		# mean and SD of MNIST - take as given
                             	])),
								batch_size=1000, shuffle=True)	
	examples = enumerate(train_loader)
	idx,  (image, label) = next(examples)


	df = pd.DataFrame(columns=['Actual Label','Label index','Predicted Label','Image Name','Confidence'])
	for i in tqdm(range(1000)):
	
		img3D = np.repeat(image[i], 3, axis=0)#convert to 3d
		batch_t = torch.unsqueeze(img3D, 0)
		model.eval()
		
		out = model(batch_t)
		
		val, index = torch.max(out, 1)
		#print(val, index.item())
		
		#plt.imshow(image[i][0])
		#plt.show()
		
		confidence = torch.nn.functional.softmax(out, dim=1)[0]*100

		df.loc[len(df)] = [label[i].item(), index.item(), labels[str(index.item())][1], 
			labels[str(index.item())][0], confidence[index].item()]
			
	return df

def plot_labels(data):
	'''
	Create plot of the results of each value 0-9 
	'''
	
	for true_label in range(10):
		truth = data['Actual Label']
		preds = np.asarray(data[truth==true_label]['Predicted Label'])
		confs = np.asarray(data[truth==true_label]['Confidence'])
	
		all_counts = {}
		all_confs = {}

		for i in range(len(preds)):
			if preds[i] not in all_counts:
				all_counts[preds[i]] = 0
				all_confs[preds[i]] = []
				
			all_counts[preds[i]] += 1
			all_confs[preds[i]].append(confs[i])

		labels = np.asarray(list(all_counts.keys()))
		counts = np.asarray(list(all_counts.values()))
		percents = np.asarray([x/sum(counts) for x in counts])
		confs = np.asarray(list(all_confs.values()))
		avg_confs = np.asarray([sum(x)/len(x) for x in confs])
		to_sort = np.argsort(counts)
		
		c = [[x/100, 0, 0] for x in avg_confs]
		
		plt.figure(figsize=[8, 5])
		plt.scatter(range(len(counts)), percents[to_sort], color=c)
		plt.xticks(range(len(labels)), labels[to_sort], rotation=90, fontsize=10)
		plt.title(true_label)
		plt.xlabel('Image Label', fontsize=12)
		plt.ylabel('Label Frequency (%)', fontsize=12)
		plt.tight_layout()
		# Add confidence numbers
		for i, txt in enumerate(avg_confs[to_sort]):
		 	plt.annotate(str(int(txt))+'%', 
		 				xy=(range(len(percents))[i], percents[to_sort][i]),
		 				xytext=(range(len(percents))[i]-0.5, percents[to_sort][i]),
		 				fontsize='small',
		 				rotation=0)


		
		#plt.show()
		

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
	
	#table = test_on_mnist(alexnet, imagenet_labels)	# Test on mnist dataset
	#table.to_csv('MNIST_results.csv')						# Save results as csv
	table = pd.read_csv('MNIST_results.csv')
	
	plot_labels(table)