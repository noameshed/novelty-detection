import csv
import json
import numpy as np
import os
import seaborn as sb

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
			


if __name__ == "__main__":
	
	# Create the category-label dictionaries
	cat_to_lab, lab_to_cat = get_categories(os.getcwd() + '/imagenet_categories.csv')
	
	# Looking at the category distribution of CNN labels for a test class
	with open(os.getcwd() + '/alexnet_inat_results/inat_results_top_choice.json') as f:
		cnn_results = json.load(f)
		organism_groups = cnn_results.keys()		# Amphibia, Fungi, Mammalia, etc.
		curclass = 'Arachnida'

		# TODO: for group in organism_groups
		# Get data on each test class
		test_classes = cnn_results[curclass].keys()
		for c in test_classes:
			# Get all of the cnn labels for that creature
			cnn_labels = cnn_results[curclass][c]['labels']
			cnn_counts = cnn_results[curclass][c]['counts']
			print(cnn_results[curclass][c])		# Keys are 'labels', 'confs', 'counts'

			# Get categories for the labels of each image in the class
			result_cats = []		# A non-unique list of the categories, e.g. [hat, but, hat, clothing]
			for r in cnn_labels:
				result_cats.append(lab_to_cat[r])

			# Make a new distribution of categories rather than labels, based on the counts
			result_cats_unique = []		# A unique list of the categories, e.g. [hat, bug, clothing]
			result_cats_count = []		# The number of occurrences of each unique category
			print(cnn_labels)
			print(result_cats)
			for i,r in enumerate(result_cats):
				# Add to unique list of categories
				if r not in result_cats_unique:
					result_cats_unique.append(r)
					result_cats_count.append(0)

				# Count how many times it appears over all images
				idx = result_cats_unique.index(r)
				count = cnn_counts[i]		# How many times has the category appeared because of this label
				result_cats_count[idx] += count

			
			# Sort in ascending order of frequency
			order = np.argsort(result_cats_count)
			result_cats_unique = np.array(result_cats_unique)[order]
			result_cats_count = np.array(result_cats_count)[order]

			# Plot the most common labels by class


			break

			

		# Plot the most common labels by animal group


