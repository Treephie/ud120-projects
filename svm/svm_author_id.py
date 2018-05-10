#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn import svm
from sklearn.metrics import accuracy_score

# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

time0 = time()
clf = svm.SVC(C=10000, kernel='rbf')
clf.fit(features_train, labels_train)
print "training time: ", time() - time0

time0 = time()
pred = clf.predict(features_test)
print "test time: ", time() - time0

accuracy = accuracy_score(labels_test, pred)
print accuracy


#########################################################

print pred[10], pred[26], pred[50]

chris_cnt = pred.tolist().count(1)
print chris_cnt
