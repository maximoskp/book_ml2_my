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

# the probabilities assigned to each oob instance can be obtained
print( repr( bag_clf.oob_decision_function_ ) )

# %% test on test data

from sklearn.metrics import accuracy_score

y_pred = bag_clf.predict( X_test )
acc = accuracy_score( y_test, y_pred)
print('acc: ', acc)

# %%  random forests

from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit( X_train, y_train )

y_pred_rf = rnd_clf.predict( X_test )
acc_rf = accuracy_score( y_test, y_pred)
print('acc_rf: ', acc_rf)

# %% same classifier as above, but not optimised
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(max_features='auto', max_leaf_nodes=16),
    n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1)

# check Extra-Trees, Extremele Random Trees, that employ random splitting 
# points for some features, instead of computing the less impure point.

# %% feature importance

# how much impurity is realted to a specific node/feature
from sklearn.datasets import load_iris

iris = load_iris()
rnd_clf = RandomForestClassifier( n_estimators=500, n_jobs=-1 )
rnd_clf.fit( iris['data'] , iris['target'] )

for name, score in zip( iris['feature_names'] , rnd_clf.feature_importances_ ):
    print( name + ': ' + str(score) )
