# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 07:41:14 2021

@author: user
"""

# %% load-split data

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X, y = make_moons( n_samples=500, noise=0.3, random_state=42 )

X_train, X_test, y_train, y_test = train_test_split( X, y, random_state=42 )

# %% create and train voting classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

voting_clf = VotingClassifier(
    estimators=[ ('lr', log_clf) , ('rf', rnd_clf) , ('svc', svm_clf) ],
    voting='hard' )
voting_clf.fit( X_train, y_train )

# voting='soft' for weighing probabilities instead of 0/1 votes for each classifier
# in this case, make sure all classifiers output probabilities instead of 0/1
# e.g. SVC produces 0/1 by default - should use predict_proba() there

# %% check accuracy of each component

from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit( X_train, y_train )
    y_pred = clf.predict( X_test )
    print( clf.__class__.__name__, accuracy_score( y_test, y_pred ) )


# %% baggging and pasting

# train the same model on different subsets of the training sets sampled:
# bagging:  with replacement
# pasting: without replacement

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1)

# bootstrap=True: bagging, i.e., with replacement
# n_jobs=-1: use all available cores

bag_clf.fit( X_train , y_train )
y_pred = bag_clf.predict(X_test)

# %% out-of-bag evaluation

# when using bootstrap (with resampling) only around 63% of the data are sampled
# the rest are oob and can be used for evaluation

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    bootstrap=True, n_jobs=-1, oob_score=True)

bag_clf.fit( X_train , y_train )
print( bag_clf.oob_score_ )

# %% page 196