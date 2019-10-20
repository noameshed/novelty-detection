# This code trains a classifier to distinguish classes that are in, not in, have relatives
# in, or have a parent in imagenet. The feature vectors are the frequencies of the top n
# labels of a particular class
import json
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def get_feature_vec(data, query, n):
	grp = query['Biological Group']
	name = query['Class']
	feat_vec = data[grp][name]['counts']

	if feat_vec is None:
		return None

	feat_vec = np.array(feat_vec)
	feat_vec /= sum(feat_vec)

	if len(feat_vec) < n:
		feat_vec = np.concatenate((np.zeros(n-len(feat_vec)), feat_vec))

	assert(len(feat_vec[-n:])==n)
	return list(feat_vec[-n:])


if __name__ == '__main__':
	# Load dataframe of inaturalist annotations
	df = pd.read_csv('in_out_class.csv')

	labeled_data = df[df['Biological Group']=='Aves']
	labeled_data = labeled_data[labeled_data['Annotator'].notnull()]

	# Collect feature vectors and labels
	n = 20
	X = []
	Y = []
	with open('inat_results.json', 'r') as f:
		f = json.load(f)
		
		for i in labeled_data.index:
			row = df.iloc[i]
			vec = get_feature_vec(f, row, n)
			if vec is None:
				continue
			X.append(list(vec))
			Y.append(row['Relation to Imagenet'])

	X = np.array(X)
	# Split the data
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

	# Train linear classifier
	clf = SVC(tol=1e-3, random_state=True)
	clf.fit(X_train, Y_train)

	print(clf.score(X_test, Y_test))

	# Train Random Forest Classifier
	clf = RandomForestClassifier(n_estimators=100)
	clf.fit(X_train, Y_train)
	print(clf.score(X_test, Y_test))

	# results = clf.predict(X_test)
	# for i, res in enumerate(results):
	# 	print(res, Y_test[i])