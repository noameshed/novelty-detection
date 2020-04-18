# Plots the distribution of class guesses for an animal instance
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
from tqdm import tqdm

def plot_all_labels(data, title=None, save=None, showplot=False):
	'''
	Creates a plot of the distribution of the top label for the images in
	the provided data file
	'''

	# Store the top class and confidence for each image
	imgs = data.keys()
	all_labels = []
	all_conf = []

	# for i in imgs:			# Do all images
	for i in np.random.choice(list(imgs), 50, replace=False):	# Select 50 random images
		try:
			toplabel = data[i]['labels'][-1]
			toplabel_conf = float(data[i]['confs'][-1])

			all_labels.append(toplabel)
			all_conf.append(toplabel_conf)
		except: 
			pass
	unique_labels = np.unique(all_labels)
	all_labels = np.array(all_labels)
	all_conf = np.array(all_conf)
	
	# Plot the chosen labels by their frequency
	label_counts = OrderedDict()
	label_conf = OrderedDict()
	for l in unique_labels:		# Count the number of times this label has been selected
		label_counts[l] = np.sum(all_labels == l)
		conf = all_conf[all_labels==l]
		avg_conf = sum(conf)/len(conf)
		label_conf[l] = avg_conf
	
	# Plot
	labels = np.array(list(label_counts.keys()))			# Labels
	count = np.array(list(label_counts.values()))		# Count of each label
	freq = count/sum(count)								# Frequency of each label
	confs = np.array(list(label_conf.values()))			# Confidence of each label
	assert(label_conf.keys() == label_counts.keys())	
	to_sort = np.argsort(count)[::-1]
	colors = [(x/100, 0, 0) for x in confs[to_sort]]

	k = kurtosis(freq[to_sort])
	H = entropy(freq[to_sort])
	title += ' (k=' + str(round(k,1)) + ', H=' + str(round(H,1)) + ')'
	plotfig(range(len(labels)), freq[to_sort], labels[to_sort], colors, title, save)

	return label_counts, label_conf, k, H

def kurtosis(data):
	"""
	Compute the kurtosis for the given distribution
	k = (SUM((Yi-Ybar)^4/N)/s^2)
	"""
	Y = np.array(data)
	s = np.std(Y)
	N = len(Y)
	Ybar = np.mean(Y)

	k = (np.sum(((Y-Ybar)**4)/N)/(s**4))
	return k

def entropy(data):
	"""
	Compute the Shannon entropy for the given distribution
	H = -sum(Pi*log2*Pi)
	"""
	H = -np.sum((data*np.log2(data)))
	return H


def plotfig(xs, ys, labels, col, title, savepath):
	if len(ys)>50:
		plt.figure(figsize=[len(xs)*.15, 7])
	else:
		plt.figure()

	ax = plt.scatter(xs, ys, color=col, zorder=2)
	plt.title(title, fontsize=14)
	plt.xlabel('Label Categories', fontsize=12)
	plt.ylabel('Label frequency', fontsize=12)

	ylocs, _ = plt.yticks()
	plt.hlines(ylocs, xmin=0, xmax=len(labels), colors='lightgrey', linestyles='dashed', zorder=1, linewidth=0.5)
	plt.xticks(np.arange(len(labels)), labels, rotation='vertical', fontsize=7)
	plt.ylim(bottom=0)
	plt.tight_layout()

	# Add confidence numbers
	# for i, txt in enumerate(confs[to_sort]):
	# 	plt.annotate(str(int(txt))+'%', 
	# 				xy=(range(len(keys))[i], vals[to_sort][i]),
	# 				xytext=(range(len(keys))[i]-0.5, vals[to_sort][i]),
	# 				fontsize='x-small',
	# 				rotation=0)


	plt.savefig(savepath, dpi=200)
	plt.close()

if __name__ == "__main__":
	savepath = os.getcwd() + '/plots_label_dist_alexnet/Aves_50picsperclass/'
	basepath = os.getcwd() + '/alexnet_inat_results/Aves/'

	total_count_dict = OrderedDict()
	total_conf_dict = OrderedDict()

	all_distros = {}
	# Plot the results from iNaturalist birds
	for fname in tqdm(os.listdir(basepath)):
		with open(basepath+fname, 'r') as f:

			f = json.load(f)

			# Optionally, skip any classes with under 50 images
			if len(f.keys())<50:
				continue

			name = fname.split('.')[0]
			all_distros[name] = {}		# Stores label distribution, kurtosis, and entropy
			title = "Label distribution, "+name

			label_counts, label_confs, k, H = plot_all_labels(f, save=savepath+'/plots/'+name+'.jpg', title=title)

			# Store information about the distribution
			all_distros[name]['labels'] = list(label_counts.keys())
			all_distros[name]['counts'] = [int(i) for i in label_counts.values()]
			all_distros[name]['confs'] = [float(i) for i in label_confs.values()]
			all_distros[name]['kurtosis'] = str(k)
			all_distros[name]['entropy'] = str(H)
			
			for l in label_counts.keys():
				if l not in total_count_dict.keys():
					total_count_dict[l] = 0
					total_conf_dict[l] = []
				total_count_dict[l] += label_counts[l]
				total_conf_dict[l].append(label_confs[l])

	# Save distribution data as json
	with open(savepath+'Aves_distros.json', 'w') as outfile:
		json.dump(all_distros, outfile)

	# Get the overall distribution for the entire category (e.g. Aves)
	total_count = list(total_count_dict.values())
	all_labels = list(total_count_dict.keys())
	all_freq = total_count/sum(total_count)
	
	# Get average confidence per label
	all_conf = []
	for c in total_conf_dict.keys():
		conflist = total_conf_dict[c]
		all_conf.append(sum(conflist)/len(conflist))
	
	# Sort by decreasing frequency
	order = np.argsort(all_freq)[::-1]
	all_freq_sorted = np.array(all_freq)[order]
	all_conf_sorted = np.array(all_conf)[order]
	all_labels_sorted = np.array(all_labels)[order]
	colors = [[x/100, 0, 0] for x in all_conf_sorted]

	# Plot the label distribution for the entire category
	plotfig(range(len(all_freq_sorted)), all_freq_sorted, all_labels_sorted, colors, title, os.getcwd()+'Aves.jpg')

