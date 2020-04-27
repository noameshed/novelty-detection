import csv
import json
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict

def get_categories(filename):
	# Open the file with list of categories and the imagenet classes
	# in each category
	# Return a dictionary of category names to the list of classes in the category
	# and a reverse lookup dictionary

	cat_to_lab = OrderedDict()
	lab_to_cat = OrderedDict()
	with open(filename) as csvfile:
		freader = csv.reader(csvfile)
		rownum = 0
		for row in freader:
			# Skip header row
			if rownum == 0:
				rownum += 1
				continue

			row = np.array(row)
			row = row[row!='']		# get rid of empty cells

			# The category is the first element in the row
			# the rest are the labels in that category
			cat = row[0]
			labs = row[1:]
			assert(1+len(labs) == len(row))

			# Store values in dictionaries if not yet there
			cat_to_lab[cat] = list(labs)
			for l in labs:
				lab_to_cat[l] = cat

	return cat_to_lab, lab_to_cat

def kurtosis(data):
	"""
	Compute the kurtosis for the given distribution
	k = (SUM((Yi-Ybar)^4/N)/s^2)
	"""
	Y = np.array(data)
	s = np.std(Y)
	N = len(Y)
	Ybar = np.mean(Y)

	k = (np.sum(((Y-Ybar)**4)/N)/(1e-15+s**4))
	return k

def entropy(data):
	"""
	Compute the Shannon entropy for the given distribution
	H = -sum(Pi*log2*Pi)
	"""
	H = -np.sum((data*np.log2(data)))
	return H

def plotfig(ys, labels, title, savepath):
	# Plot the most common labels by category
	if len(ys)>50:
		plt.figure(figsize=[10,7])
	elif len(ys)>75:
		plt.figure(figsize=[13,7])
	else:
		plt.figure()

	ax = sns.scatterplot(range(len(ys)), ys, zorder=2)
	plt.title(title, fontsize=14)
	plt.xlabel('Label Categories', fontsize=12)
	plt.ylabel('Label frequency', fontsize=12)
	
	plt.ylim(bottom=0, top=1.)
	ylocs, _ = plt.yticks()
	plt.hlines(np.arange(0,1.0, 0.1), xmin=0, xmax=len(labels)-1, colors='lightgrey', linestyles='dashed', zorder=1, linewidth=0.5)
	plt.xticks(range(len(ys)), labels, rotation='vertical', fontsize=10)
	plt.tight_layout()

	#plt.show()
	# Save resulting plot
	fig = ax.get_figure()
	fig.savefig(savepath, dpi=200)
	plt.close()

if __name__ == "__main__":
	
	# Create the category-label dictionaries
	cat_to_lab, lab_to_cat = get_categories(os.getcwd() + '/imagenet_categories.csv')
	all_cats = list(cat_to_lab.keys())
	total_count = np.zeros(len(all_cats))

	savepath = os.getcwd() + '/plots_category_dist_alexnet/'

	# Looking at the category distribution of CNN labels for a test class
	with open(os.getcwd() + '/alexnet_inat_results/inat_results_top_choice.json') as f:
		cnn_results = json.load(f)
		organism_groups = cnn_results.keys()		# Amphibia, Fungi, Mammalia, etc.
		for curclass in organism_groups:
			# Create save folder for the organism
			try:
				os.mkdir(savepath + curclass)
			except:
				pass

			if curclass != 'Aves':
				continue
			
			all_distros = {}		# Stores category distribution, kurtosis, and entropy
			# Get data on each test class
			test_classes = cnn_results[curclass].keys()
			for c in test_classes:
				# Get all of the cnn labels for that creature
				cnn_labels = cnn_results[curclass][c]['labels']
				cnn_counts = cnn_results[curclass][c]['counts']

				# Get categories for the labels of each image in the class
				result_cats = []		# A non-unique list of the categories, e.g. [hat, bug, hat, clothing]
				if cnn_labels is None: #or len(cnn_labels) < 50:		# Optionally, skip any classes with under 50 images
					continue
				for r in cnn_labels:
					result_cats.append(lab_to_cat[r])

				# Make a new distribution of categories rather than labels, based on the counts
				result_cats_labels = []		# A unique list of the categories, e.g. [hat, bug, clothing]
				result_cats_count = []		# The number of occurrences of each unique category

				for i,r in enumerate(result_cats):	
				# for i, r in enumerate(np.random.choice(result_cats, 50, replace=False)):	# select 50 random images
					# Add to unique list of categories
					if r not in result_cats_labels:
						result_cats_labels.append(r)
						result_cats_count.append(0)

					# Count how many times it appears over all images
					idx = result_cats_labels.index(r)
					count = cnn_counts[i]		# How many times has the category appeared because of this label
					result_cats_count[idx] += count

					# Add that count to the total count for this organism group (i.e. Aves)
					idx = all_cats.index(r)
					total_count[idx] += count
				
				# Sort in ascending order of frequency
				order = np.argsort(result_cats_count)[::-1]
				result_cats_labels = np.array(result_cats_labels)[order]
				result_cats_count = np.array(result_cats_count)[order]
				result_cats_freq = result_cats_count/np.sum(result_cats_count)

				# Calculate kurtosis and entropy
				k = kurtosis(list(result_cats_freq))
				H = entropy(result_cats_freq)
				title = 'Test Class: '+c #+ ' (k=' + str(round(k,1)) + ', H=' + str(round(H,1)) + ')'
				plotfig(result_cats_freq, result_cats_labels, title, savepath+curclass+'/'+c+'.png')

				# Store information on kurtosis, entropy, and categories
				all_distros[c] = {}
				all_distros[c]['labels'] = list(result_cats_labels)
				all_distros[c]['counts'] = [int(i) for i in result_cats_count]
				all_distros[c]['kurtosis'] = str(k)
				all_distros[c]['entropy'] = str(H)

			# Plot the most common labels by animal group
			total_freq = total_count/sum(total_count)
			order = np.argsort(total_freq)[::-1]
			total_freq_sorted = np.array(total_freq)[order]
			all_cats_sorted = np.array(all_cats)[order]
			title = 'Average categories: '+curclass
			plotfig(total_freq_sorted, all_cats_sorted, title, savepath+curclass+'.png')

			# Save distribution data as json
			with open(savepath+curclass+'_distros.json', 'w') as outfile:
				json.dump(all_distros, outfile)


