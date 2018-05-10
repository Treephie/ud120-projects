#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus', 'from_this_person_to_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL")
### Task 3: Create new feature(s)
# def poi_email_ratio(from_poi_to_this_person, to_messages):
#     if from_poi_to_this_person or to_messages == 'NaN':
#         to_poi_ratio = 0
#     else:
#         to_poi_ratio = float(from_poi_to_this_person)/to_messages
#     return to_poi_ratio
#
# # create new key and value
# for key in data_dict:
#     data_dict[key]['to_poi_ratio'] = poi_email_ratio(data_dict[key]['from_poi_to_this_person'],  data_dict[key]['to_messages'])
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# feature scaling
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(features)
# scaler.transform(features)
#
import numpy as np
features = np.array(features)
labels = np.array(labels)
#
print features.shape, labels.shape
# plot
import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame(features, columns=['salary', 'bonus', 'from_this_person_to_poi'])
pd.tools.plotting.scatter_matrix(df, alpha=0.3, diagonal='kde')
plt.show()

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn import tree
from sklearn.grid_search import GridSearchCV
trees = tree.DecisionTreeClassifier()
parameters = {'min_samples_split' : range(5,80,5), 'splitter' : ('best', 'random')}

clf = GridSearchCV(trees, parameters)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf.fit(features_train, labels_train)
print clf.score(features_test, labels_test)

# pca
# from sklearn.decomposition import RandomizedPCA
# pca = RandomizedPCA(n_components=3, whiten=True).fit(features_train)
# print "features.shape", features_train.shape
# print "pca.componnets_:", pca.components_
# print "pca.explained_variance_:", pca.explained_variance_
#
# features_train_pca = pca.transform(features_train)
# features_test_pca = pca.transform(features_test)
#
#
# clf.fit(features_train_pca, labels_train)
# pred = clf.predict(features_test_pca)
# from sklearn.metrics import accuracy_score
# print "accuracy score:", accuracy_score(pred, labels_test)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


#
# test_color = 'r'
# train_color = 'b'
#
# ### draw the scatterplot, with color-coded training and testing points
# import matplotlib.pyplot as plt
# for feature, target in zip(features_test, labels_test):
#     plt.scatter( feature, target, color=test_color)
# for feature, target in zip(features_train, labels_train):
#     plt.scatter( feature, target, color=train_color)
#
# ### labels for the legend
# plt.scatter(features_test[0], labels_test[0], color=test_color, label="test")
# plt.scatter(features_test[0], labels_test[0], color=train_color, label="train")
#
# ### draw the regression line, once it's coded
# # try:
# #     plt.plot( features_test, pred)
# # except NameError:
# #     pass
#
# # plt.plot(feature_train, reg.predict(feature_train), color="b")
# # print "new slope:", reg.coef_
#
#
# plt.xlabel(features_list[1])
# plt.ylabel(features_list[0])
# plt.legend()
# plt.show()
