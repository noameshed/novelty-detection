# This code trains a classifier to distinguish classes that are in, not in, have relatives
# in, or have a parent in imagenet. The feature vectors are the frequencies of the top n
# labels of a particular class
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

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

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    classes = np.unique(y_true)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax

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
	clf = SVC(tol=1e-3, random_state=True, gamma='auto')
	clf.fit(X_train, Y_train)
	preds = clf.predict(X_test)
	for p in preds:
		print(p)
	print('SVM:', clf.score(X_test, Y_test))
	classes = ['relative in imagenet', 'in imagenet', 'parent in imagenet', 'not in imagenet']
	plot_confusion_matrix(Y_test, preds, classes, normalize=False)

	# Train Random Forest Classifier
	clf = RandomForestClassifier(n_estimators=100)
	clf.fit(X_train, Y_train)
	print('Random Forest:', clf.score(X_test, Y_test))

	results = clf.predict(X_test)
	for i, res in enumerate(results):
		print(res, Y_test[i])