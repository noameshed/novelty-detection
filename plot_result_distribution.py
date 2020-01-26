# Plots the distribution of class guesses for an animal instance
import json
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_all_labels(data, title=None, save=None, showplot=False):
	'''
	Creates a plot of the distribution of the top label for the images in
	the provided data file
	'''

	# Store the top class and confidence for each image
	imgs = data.keys()
	all_labels = []
	all_conf = []
	for i in imgs:
		toplabel = data[i]['labels'][-1]
		toplabel_conf = float(data[i]['confs'][-1])

		all_labels.append(toplabel)
		all_conf.append(toplabel_conf)
	unique_labels = np.unique(all_labels)
	all_labels = np.array(all_labels)
	all_conf = np.array(all_conf)
	
	# Plot the chosen labels by their frequency
	label_counts = {}
	label_conf = {}
	for l in unique_labels:		# Count the number of times this label has been selected
		label_counts[l] = np.sum(all_labels == l)
		conf = all_conf[all_labels==l]
		avg_conf = sum(conf)/len(conf)
		label_conf[l] = avg_conf

	# Plot
	keys = np.array(list(label_counts.keys()))			# Labels
	vals = np.array(list(label_counts.values()))		# Count of each label
	freq = vals/sum(vals)												# Frequency of each label
	confs = np.array(list(label_conf.values()))			# Confidence of each label
	assert(label_conf.keys() == label_counts.keys())	
	to_sort = np.argsort(vals)
	colors = [[x/100, 0, 0] for x in confs[to_sort]]

	if len(to_sort) > 50:
		plt.figure(figsize=[20,8])
	else:
		plt.figure()
	plt.ylim((0,1.1))
	plt.scatter(range(len(keys)), freq[to_sort], c=colors)
	plt.xticks(np.arange(len(keys)), keys[to_sort], rotation=90, fontsize=10)
	if title is not None:
		plt.title(title)
	plt.xlabel('Image Label', fontsize=14)
	plt.ylabel('Label frequency', fontsize=14)
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


if __name__ == "__main__":
	path = os.getcwd() + '/alexnet_birdvid_results/'
	savepath = 'C:/Users/noam_/Documents/Cornell/CS7999/1_27_20/inaturalist_label_distributions/'

	# To plot the video results
	# for species in os.listdir(path):
	# 	# Load the results of testing the video frames
	# 	for results in os.listdir(path + species + '/'):
	# 		with open(path + species + '/' + results, 'r') as f:
	# 			f = json.load(f)
	# 			# for each video or set of images, plot the distribution of top 1 labels
	# 			title = "Label distribution for "+species
				
	# 			plot_all_labels(f, save=savepath+species+'.jpg', title=title)

	# Plot the results from iNaturalist only of the birds also in the videos
	with open('vids_to_split.txt', 'r') as f:
		i=1
		for line in f:
			splitline = line.strip().split('\t')
			ID = int(splitline[0])
			latin_name = splitline[1]
			name = splitline[2]

			files = os.listdir(os.getcwd()+'/alexnet_inat_results/Aves/')

			with open(os.getcwd()+'/alexnet_inat_results/Aves/'+latin_name+'.json', 'r') as f:
				f = json.load(f)
				title = "Label distribution for "+name
				plot_all_labels(f, save=savepath+name+'.jpg', title=title)
			
			

