import json
import numpy as np
import os
import pandas as pd




if __name__ == '__main__':

	# Load dataframe of inaturalist annotations
	df = pd.read_csv('in_out_class.csv')
	

	# Extract only Aves that are labeled
	birds = df[df['Biological Group']=='Aves']
	birds = birds[birds['Annotator'].notnull()]
	print(len(birds), ' total annotated birds')
	
	rels = birds[birds['Relation to Imagenet']=='relative in imagenet']
	print(len(rels), ' with relatives in imagenet')

	is_in = birds[birds['Relation to Imagenet']=='in imagenet']
	print(len(is_in), ' in imagenet')

	par = birds[birds['Relation to Imagenet']=='parent in imagenet']
	print(len(par), ' parent in imagenet')

	not_in = birds[birds['Relation to Imagenet']=='not in imagenet']
	print(len(not_in), ' not in imagenet')
