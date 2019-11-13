import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


def find(dic, query):
	grp = query['Biological Group']
	name = query['Class']
	return dic[grp][name]

def n_most_common(data, n):
	'''
	Returns the top n most common labels in the data, along with their associated
	confidence levels, counts, and percentages
	'''

	top = {}
	top['labels'] = np.chararray(n)
	top['confs'] = np.zeros(n)
	top['counts'] = np.zeros(n)
	top['percents'] = np.zeros(n)

	n = min(n, len(data['labels']))
	
	percents = [x/sum(data['counts']) for x in data['counts']]

	top['labels'][-n:] = data['labels'][-n:]
	top['confs'][-n:] = data['confs'][-n:]
	top['counts'][-n:] = data['counts'][-n:]
	top['percents'][-n:] = percents[-n:]

	return top

def avg_distribution(data, n, dic):
	'''
	This function averages the distribution of the top n most common
	labels for each class in the given data set
	'''
	class_confs = np.array([]).reshape(0,n)
	class_freq = np.array([]).reshape(0,n)

	for idx in data.index:
		row = df.iloc[idx]
		results = find(dic, row)

		if results['labels'] is not None:
			# Add the top labels to the distribution
			top = n_most_common(results,n)
			class_confs = np.vstack((class_confs, top['confs']))
			class_freq = np.vstack((class_freq, top['percents']))

	# Get stdev for confidence and frequency of top labels
	std_confs = np.std(class_confs, axis=0)
	std_freq = np.std(class_freq, axis=0)

	# Get average confidence and frequency of top labels
	class_confs = np.sum(class_confs, axis=0)/class_confs.shape[0]
	class_freq = np.sum(class_freq, axis=0)/class_freq.shape[0]

	return class_confs, class_freq, std_confs, std_freq

def print_error_stats(err, avgs, n):
	mins = err[0,:]
	maxs = err[1,:]
	diff = maxs - mins
	
	to_print = 'Min:\t'
	for m in mins:
		to_print += str(round(m, 6))+'\t'
	print(to_print)

	to_print = 'Max:\t'
	for m in maxs:
		to_print += str(round(m, 6))+'\t'
	print(to_print)

	to_print = 'Diff:\t'
	for m in diff:
		to_print +=str(round(m, 6))+'\t'
	print(to_print)
		
	to_print = 'Avg:\t'
	for m in avgs:
		to_print += str(round(m, 6))+'\t'
	print(to_print)


if __name__ == '__main__':

	# Load dataframe of inaturalist annotations
	df = pd.read_csv('in_out_class.csv')
	
	# Extract only Aves that are labeled
	birds = df[df['Biological Group']=='Aves']
	fish = df[df['Biological Group']=='Actinopterygii']
	animalia = df[df['Biological Group']=='Animalia']
	
	# labeled_data = pd.concat([birds, fish, animalia])
	labeled_data = birds
	print(labeled_data.head(n=5))	

	labeled_data = labeled_data[labeled_data['Annotator'].notnull()]
	

	# Split into four categories
	rels = labeled_data[labeled_data['Relation to Imagenet']=='relative in imagenet']
	is_in = labeled_data[labeled_data['Relation to Imagenet']=='in imagenet']
	par = labeled_data[labeled_data['Relation to Imagenet']=='parent in imagenet']
	not_in = labeled_data[labeled_data['Relation to Imagenet']=='not in imagenet']
	
	# Split into 2 categories - is_in includes parents and relatives
	# is_in = pd.concat([is_in, par, rels], axis=0)
	
	# Split into 2 categories: not_in includes parents and relatives
	# not_in = pd.concat([not_in, par, rels], axis=0)

	# Split into 3 categories: rels includes parents and relatives
	# rels = pd.concat([rels, par], axis=0)

	# Print count statistics
	print(len(labeled_data), '\ttotal annotated')
	print(len(rels), '\twith relatives in imagenet')
	print(len(is_in), '\tin imagenet')
	print(len(par), '\tparent in imagenet')
	print(len(not_in), '\tnot in imagenet')

	# inat_results_top_choice.json saves the top result for each image in each class (in each biological group)
	with open('alexnet_inat_results/inat_results_top_choice.json', 'r') as f:
		f = json.load(f)
		n = 10
		# Set up plot
		plt.figure()
		plt.xticks(np.arange(0,-1))
		message = 'Top ' + str(n) + ' labels (' + str(n) + 'th to 1st most common)'
		plt.xlabel(message)
		plt.ylabel('Label Frequency (%)')
		plt.title('Birds: Label frequency for in vs out class')

		# Plot the distribution for the top n labels split by group (in, not in, par, rel)
		for data in [is_in, not_in, rels, par]:
			class_confs, class_freq, std_confs, std_freq = avg_distribution(data, n, f)	
			plt.errorbar(np.arange(n), class_freq, std_freq, elinewidth=0.5, capsize = 2)

		plt.legend(['In Imagenet','Not In Imagenet','Relative In Imagenet','Parent In Imagenet'], loc='upper left')
		# plt.legend(['In Imagenet','Not In Imagenet', 'Parents + Relatives'], loc='upper left')
		plt.show()