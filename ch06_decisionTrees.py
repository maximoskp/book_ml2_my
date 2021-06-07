# -*- coding: utf-8 -*-
"""
Created on Fri May 21 22:14:46 2021

@author: user
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data[:,2:] # pedal length and width
y = iris.target

tree_clf = DecisionTreeClassifier( max_depth=2 )
tree_clf.fit( X, y )

# %% export tree visualisation

from sklearn.tree import export_graphviz

export_graphviz(
    tree_clf,
    out_file = 'figs/ch6_iris_tree.dot',
    feature_names = iris.feature_names[2:],
    class_names = iris.target_names,
    rounded = True,
    filled = True
)

# transform .dot to .png
import os
os.system('dot -Tpng figs/ch6_iris_tree.dot -o figs/ch6_iris_tree.png')

# %% make prediction
# traverse the tree and get the number of instances per class in the leaf node
# ratio of instances from the asked class is the probability that the
# predicted instance belongs to this class

tmp_prediction_proba = tree_clf.predict_proba( [ [5, 1.5] ] )
print( 'tmp_prediction_proba: ', tmp_prediction_proba )

tmp_prediction = tree_clf.predict( [ [5,1.5] ] )
print( 'tmp_prediction: ', tmp_prediction )

# however, probabilities are still the same even if input might be more clear
tmp_prediction_proba = tree_clf.predict_proba( [ [6, 1.5] ] )
print( 'tmp_prediction_proba: ', tmp_prediction_proba )

# Decision trees for regression: p183
# Sensitivity to orientation: helps to use PCA p185