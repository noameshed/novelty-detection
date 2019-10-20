import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


def find(data, query):
	grp = query['Biological Group']
	name = query['Class']
	return data[grp][name]

def top_n(data, n):

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

def get_results(data, n):
	# Get results for in-class
	class_confs = np.zeros(n)
	class_per = np.zeros(n)
	for i in data.index:
		row = df.iloc[i]
		results = find(f, row)
		if results['labels'] is not None:
			top = top_n(results,n)
			class_confs += top['confs']
			class_per += top['percents']

	class_confs /= len(data.index)
	class_per /= len(data.index)		

	return class_confs, class_per


if __name__ == '__main__':

	# Load dataframe of inaturalist annotations
	df = pd.read_csv('in_out_class.csv')
	

	# Extract only Aves that are labeled
	birds = df[df['Biological Group']=='Aves']
	fish = df[df['Biological Group']=='Actinopterygii']
	animalia = df[df['Biological Group']=='Animalia']
	
	labeled_data = pd.concat([birds, fish, animalia])

	labeled_data = labeled_data[labeled_data['Annotator'].notnull()]
	print(len(labeled_data), ' total annotated')
	
	rels = labeled_data[labeled_data['Relation to Imagenet']=='relative in imagenet']
	print(len(rels), ' with relatives in imagenet')

	is_in = labeled_data[labeled_data['Relation to Imagenet']=='in imagenet']
	print(len(is_in), ' in imagenet')

	par = labeled_data[labeled_data['Relation to Imagenet']=='parent in imagenet']
	print(len(par), ' parent in imagenet')

	not_in = labeled_data[labeled_data['Relation to Imagenet']=='not in imagenet']
	print(len(not_in), ' not in imagenet')


	with open('inat_results.json', 'r') as f:
		f = json.load(f)
		n = 20
		# Set up plot
		plt.figure()
		plt.xticks(np.arange(0,-1))
		message = 'Top ' + str(n) + ' labels (' + str(n) + 'th to 1st most common)'
		plt.xlabel(message)
		plt.ylabel('Label Frequency (%)')
		plt.title('All Data: Label frequency for in vs out class')

		# Get results for in-class
		in_class_confs, in_class_per = get_results(is_in, n)	
		plt.plot(np.arange(n), in_class_per)
		
		# Get results for not in class
		out_class_confs, out_class_per = get_results(not_in, n)	
		plt.plot(np.arange(n), out_class_per)

		# Get results for relative in class
		rel_class_confs, rel_class_per = get_results(rels, n)	
		plt.plot(np.arange(n), rel_class_per)

		# Get results for parent in class
		par_class_confs, par_class_per = get_results(par, n)	
		plt.plot(np.arange(n), par_class_per)

		plt.legend(['In Imagenet','Not In Imagenet','Relative In Imagenet','Parent In Imagenet'])
		plt.show()