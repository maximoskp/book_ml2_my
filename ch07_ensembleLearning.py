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

# %% adaboost

from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm='SAMME.R', learning_rate=0.5)
ada_clf.fit(X_train, y_train)

# %% gradient boost

# make random quadratic
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3*X[:,0]**2 + 0.05*np.random.randn(100)

plt.plot(X, y, '.')

# %%

from sklearn.tree import DecisionTreeRegressor

tree_reg1 = DecisionTreeRegressor(max_depth=2)
tree_reg1.fit( X , y )

# %% predict, keep training set errors and retrain with errors
y_pred_1 = tree_reg1.predict( X )
y2 = y - y_pred_1

'''
plt.clf()
plt.plot(X,y, '.')
plt.plot(X,y2, 'r.')
plt.show()
'''

tree_reg2 = DecisionTreeRegressor(max_depth=2)
tree_reg2.fit( X , y2 )

# once more
y_pred_2 = tree_reg2.predict( X )
y3 = y2 - y_pred_2
tree_reg3 = DecisionTreeRegressor(max_depth=2)
tree_reg3.fit( X , y3 )

y_pred_3 = tree_reg3.predict(X)

plt.clf()
plt.subplot(321)
plt.plot(X,y,'.')
plt.plot(X,y_pred_1,'g.')
plt.subplot(323)
plt.plot(X,y2,'.')
plt.plot(X,y_pred_2,'g.')
plt.subplot(325)
plt.plot(X,y3,'.')
plt.plot(X,y_pred_3,'g.')
plt.subplot(322)
plt.plot(X,y,'.')
plt.plot(X,y_pred_1,'r.')
plt.subplot(324)
plt.plot(X,y,'.')
plt.plot(X,y_pred_1+y_pred_2,'r.')
plt.subplot(326)
plt.plot(X,y,'.')
plt.plot(X,y_pred_1+y_pred_2+y_pred_3,'r.')
plt.show()

# %% all the above done by gradient boosting

from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor( max_depth=2, n_estimators=3, learning_rate=1.0 )
gbrt.fit( X , y )

y_pred_gbrt = gbrt.predict( X )

plt.clf()
plt.plot(X,y,'.')
plt.plot( X , y_pred_gbrt , 'r.' )
plt.show()

# %% overfitting: too many predictors

gbrt = GradientBoostingRegressor( max_depth=2, n_estimators=30, learning_rate=1.0 )
gbrt.fit( X , y )

y_pred_gbrt = gbrt.predict( X )

plt.clf()
plt.plot(X,y,'.')
plt.plot( X , y_pred_gbrt , 'r.' )
plt.show()

# %% staged prediction for getting the best number of estimators

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_val, y_train, y_val = train_test_split(X, y)

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators = 120)
gbrt.fit(X_train, y_train)

errors = [mean_squared_error(y_val, y_pred) for y_pred in gbrt.staged_predict(X_val)]

bst_n_estimators = np.argmin(errors) + 1

gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators)
gbrt_best.fit(X_train, y_train)
y_pred_best = gbrt_best.predict( X )

plt.clf()
plt.subplot(121)
plt.plot( np.arange(120), errors )
plt.plot( bst_n_estimators-1, errors[bst_n_estimators], 'xr' )
plt.subplot(122)
plt.plot(X,y,'.')
plt.plot( X , y_pred_best , 'r.' )
plt.show()

# %% early stopping when error goes up N times

gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True, subsample=0.25)
# warm_start=True keeps trees as added during learning: incremental learning
# subsample: the ratio of data to be used in training

min_val_error = float('inf')
error_going_up = 0
final_n_estimators = 0
for n_estimators in range(1, 120):
    final_n_estimators = n_estimators
    gbrt.n_estimators = n_estimators
    gbrt.fit( X_train, y_train )
    y_pred = gbrt.predict( X_val )
    val_error = mean_squared_error(y_val, y_pred)
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up += 1
        if error_going_up >= 5:
            # final_n_estimators = n_estimators - 5 # my comment
            break # early stopping

print('n_estimators = ', str(final_n_estimators))

# %% xgboost

import xgboost

xgb_reg = xgboost.XGBRegressor()
xgb_reg.fit(X_train, y_train)
y_pred = xgb_reg.predict(X_val)

# %% early stopping inherent

xgb_reg.fit(X_train, y_train,
            eval_set=[(X_val, y_val)], early_stopping_rounds=2)
y_pred = xgb_reg.predict(X_val)