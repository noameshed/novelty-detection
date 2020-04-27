import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt


def stat_avgs(species, datafile):
	"""
	Computes average kurtosis and entropy for the list of species
	"""
	k = []
	H = []
	for s in species:
		# If there were not enough samples, the species may not have associated stats
		if s not in datafile:
			continue

		if not np.isnan(float(datafile[s]['kurtosis'])):
			k.append(float(datafile[s]['kurtosis']))
			H.append(float(datafile[s]['entropy']))

	k_avg = np.mean(k)
	H_avg = np.mean(H)

	# Compute standard deviation
	k_std = np.std(k)
	H_std = np.std(H)

	return k_avg, H_avg, k_std, H_std

if __name__ == "__main__":
	# Load in-out class data
	data = pd.read_csv('in_out_class.csv')
	birds = data[data['Biological Group']=='Aves']
	
	lab_birds = birds[birds['Annotator'].notnull()]
	
	# Extract ImageNet relationships of labelled birds
	rels = lab_birds[lab_birds['Relation to Imagenet']=='relative in imagenet']['Class']
	is_in = lab_birds[lab_birds['Relation to Imagenet']=='in imagenet']['Class']
	par = lab_birds[lab_birds['Relation to Imagenet']=='parent in imagenet']['Class']
	not_in = lab_birds[lab_birds['Relation to Imagenet']=='not in imagenet']['Class']

	# Load data with kurtosis and entropy values
	fname = os.getcwd() + '/plots_category_dist_alexnet/Aves_distros.json'
	with open(fname, 'r') as f:
		f = json.load(f)

		# Get average kurtosis and entropy for each of the 4 categories
		rels_k, rels_h, rels_k_std, rels_h_std = stat_avgs(rels, f)
		in_k, in_h, in_k_std, in_h_std = stat_avgs(is_in, f)
		par_k, par_h, par_k_std, par_h_std = stat_avgs(par, f)
		not_k, not_h, not_k_std, not_h_std = stat_avgs(not_in, f)

	# Organize data for plot
	xs = np.arange(0, 4.)
	labels = ['In ImageNet', 'Not In ImageNet', 'Parent In ImageNet', 'Relative In ImageNet']
	ks = [in_k, not_k, par_k, rels_k]
	Hs = [in_h, not_h, par_h, rels_h]
	ks_std = [in_k_std, not_k_std, par_k_std, rels_k_std]
	Hs_std = [in_h_std, not_h_std, par_h_std, rels_h_std]

	print(ks)
	# Plot results - kurtosis
	plt.figure()
	w = 0.2
	plt.bar(xs-w/4, ks, width=w, label='kurtosis', color='red')
	plt.errorbar(xs-w/4, ks, yerr=ks_std, linewidth=0, elinewidth=1, color='black', capsize=2)

	print(ks_std)
	plt.title('Aves Average Kurtosis by Category', fontsize=14)
	plt.xlabel('Label Categories', fontsize=12)
	plt.ylabel('Kurtosis Value', fontsize=12)
	
	ylocs, _ = plt.yticks()
	plt.hlines(ylocs, xmin=0, xmax=len(xs)-1, colors='lightgrey', linestyles='dashed', zorder=1, linewidth=0.5)
	plt.xticks(np.arange(len(labels)), labels, fontsize=9)
	plt.tight_layout()
	plt.show()

	# plot results - entropy
	plt.figure()
	w = 0.2
	plt.bar(xs-w/4, Hs, width=w, label='entropy', color='red')
	plt.errorbar(xs-w/4, Hs, yerr=Hs_std, linewidth=0, elinewidth=1, color='black', capsize=2)

	plt.title('Aves Average Entropy by Category', fontsize=14)
	plt.xlabel('Label Categories', fontsize=12)
	plt.ylabel('Entropy Value', fontsize=12)
	

	ylocs, _ = plt.yticks()
	plt.hlines(ylocs, xmin=0, xmax=len(xs)-1, colors='lightgrey', linestyles='dashed', zorder=1, linewidth=0.5)
	plt.xticks(np.arange(len(labels)), labels,  fontsize=9)
	plt.tight_layout()
	plt.show()

