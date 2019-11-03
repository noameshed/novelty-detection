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

def get_results(data, n, dic):
	# Get results for in-class
	class_confs = np.zeros(n)
	class_per = np.zeros(n)
	err_per = np.zeros((2, n))
	err_per[0,:] = np.full((1,n), np.inf)
	err_confs = np.zeros((2, n))
	err_confs[0,:] = np.full((1,n), np.inf)

	for idx in data.index:
		row = df.iloc[idx]
		results = find(dic, row)
		if results['labels'] is not None:

			top = n_most_common(results,n)
			class_confs += top['confs']
			class_per += top['percents']

			idx_min = top['percents']<err_per[0,:]
			idx_max = top['percents']>err_per[1,:]
			err_per[0,:][idx_min] = top['percents'][idx_min]
			err_per[1,:][idx_max] = top['percents'][idx_max]
			
			idx_min = top['confs']<err_conf[0,:]
			idx_max = top['confs']>err_conf[1,:]
			err_confs[0,:][idx_min] = top['confs'][idx_min]
			err_confs[1,:][idx_max] = top['confs'][idx_max]
			
	# Get average confidence and frequency
	class_confs /= len(data.index)
	class_per /= len(data.index)		

	print_error_stats(err_confs, class_confs, n)

	# Normalize the errors to be centered at the average
	err_per[0,:] = class_per - err_per[0,:]
	err_per[1,:] = err_per[1,:] - class_per

	err_conf[0,:] = class_confs - err_conf[0,:]
	err_conf[1,:] = err_conf[1,:] - class_confs

	return class_confs, class_per, err_per, err_conf

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

	# inat_results.json saves the top result for each image in each class (in each biological group)
	with open('inat_results.json', 'r') as f:
		f = json.load(f)
		n = 20
		# Set up plot
		plt.figure()
		plt.xticks(np.arange(0,-1))
		message = 'Top ' + str(n) + ' labels (' + str(n) + 'th to 1st most common)'
		plt.xlabel(message)
		plt.ylabel('Label Confidence (%)')
		plt.title('Birds: Label confidence for in vs out class')

		# Get results for in-class
		in_class_confs, in_class_per, err_freq, err_conf = get_results(is_in, n, f)	
		plt.errorbar(np.arange(n), in_class_confs, err_conf, elinewidth=0.5, capsize = 2)
		
		# Get results for not in class
		out_class_confs, out_class_per, err_freq, err_conf = get_results(not_in, n, f)	
		plt.errorbar(np.arange(n), out_class_confs, err_conf, elinewidth=0.5, capsize = 2)

		# Get results for relative in class
		rel_class_confs, rel_class_per, err_freq, err_conf = get_results(rels, n, f)	
		plt.errorbar(np.arange(n), rel_class_confs, err_conf, elinewidth=0.5, capsize = 2)

		# Get results for parent in class
		par_class_confs, par_class_per, err_freq, err_conf = get_results(par, n, f)	
		plt.errorbar(np.arange(n), par_class_confs, err_conf, elinewidth=0.5, capsize = 2)

		plt.legend(['In Imagenet','Not In Imagenet','Relative In Imagenet','Parent In Imagenet'], loc='upper left')
		# plt.legend(['In Imagenet','Not In Imagenet', 'Parents + Relatives'], loc='upper left')
		plt.show()