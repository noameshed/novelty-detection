import csv
import json
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

def get_categories(filename):
	# Open the file with list of categories and the imagenet classes
	# in each category
	# Return a dictionary of category names to the list of classes in the category
	# and a reverse lookup dictionary

	cat_to_lab = {}
	lab_to_cat = {}
	# TODO: Assert that there are 1000 classes
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
			
def plotfig(ys, labels, title, savepath):
	# Plot the most common labels by class
	plt.figure()
	ax = sns.scatterplot(range(len(ys)), ys)
	plt.title(title)
	plt.xlabel('Label Categories')
	plt.ylabel('Label frequency')
	plt.xticks(range(len(ys)))
	ax.set_xticklabels(labels, rotation='vertical', fontsize=8)
	plt.tight_layout()

	#plt.show()
	# Save resulting plot
	fig = ax.get_figure()
	fig.savefig(savepath)
	plt.close()

if __name__ == "__main__":
	
	# Create the category-label dictionaries
	cat_to_lab, lab_to_cat = get_categories(os.getcwd() + '/imagenet_categories.csv')
	savepath = os.getcwd() + '/plots_category_dist/'

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

			all_cats = list(cat_to_lab.keys())
			total_count = np.zeros(len(all_cats))

			# Get data on each test class
			test_classes = cnn_results[curclass].keys()
			for c in test_classes:
				# Get all of the cnn labels for that creature
				cnn_labels = cnn_results[curclass][c]['labels']
				cnn_counts = cnn_results[curclass][c]['counts']

				# Get categories for the labels of each image in the class
				result_cats = []		# A non-unique list of the categories, e.g. [hat, but, hat, clothing]
				if cnn_labels is None:
					continue
				for r in cnn_labels:
					result_cats.append(lab_to_cat[r])

				# Make a new distribution of categories rather than labels, based on the counts
				result_cats_unique = []		# A unique list of the categories, e.g. [hat, bug, clothing]
				result_cats_count = []		# The number of occurrences of each unique category

				for i,r in enumerate(result_cats):					
					# Add to unique list of categories
					if r not in result_cats_unique:
						result_cats_unique.append(r)
						result_cats_count.append(0)

					# Count how many times it appears over all images
					idx = result_cats_unique.index(r)
					count = cnn_counts[i]		# How many times has the category appeared because of this label
					result_cats_count[idx] += count

					# Add that count to the total count for this organism group (i.e. Aves)
					idx = all_cats.index(r)
					total_count[idx] += count
				
				# Sort in ascending order of frequency
				order = np.argsort(result_cats_count)
				result_cats_unique = np.array(result_cats_unique)[order]
				result_cats_count = np.array(result_cats_count)[order]
				result_cats_freq = result_cats_count/sum(result_cats_count)

				title = 'Test Class: '+c
				plotfig(result_cats_freq, result_cats_unique, title, savepath+curclass+'/'+c+'.png')

			# Plot the most common labels by animal group
			total_freq = total_count/sum(total_count)
			order = np.argsort(total_freq)
			total_freq_sorted = np.array(total_freq)[order]
			all_cats_sorted = np.array(all_cats)[order]
			title = 'Average categories: '+curclass
			plotfig(total_freq_sorted, all_cats_sorted, title, savepath+curclass+'.png')


