# -*- coding: utf-8 -*-
"""
Created on Sun May  2 08:23:39 2021

@author: user
"""

import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()

X = iris['data'][:, (2,3)]
y = (iris['target'] == 2).astype( np.float64 )

# %%

svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_svc', LinearSVC( C=1, loss='hinge' ))
])

# C cloaser to 1 allows more classification violations for larger
# support vector margins

svm_clf.fit( X , y )

y_predict = svm_clf.predict( [ [5.5,1.7] ] )
print('y_predict: ' + repr(y_predict))
# SVMs do not provide probability estimation

# %% polynomial fit - increasing dimensions

from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

X, y = make_moons(n_samples=100, noise=0.15)

# %% 
polynomial_svm_clf = Pipeline([
    ('poly_features', PolynomialFeatures(degree=3)),
    ('scaler', StandardScaler()),
    ('svm_clf', LinearSVC(C=10, loss='hinge'))
])

polynomial_svm_clf.fit(X, y)

# %% polynomial kernel

from sklearn.svm import SVC

poly_kernel_svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_clf', SVC(kernel='poly', degree=3, coef0=1, C=5))
])
# coef0: influence of high-degree polynomials
poly_kernel_svm_clf.fit( X , y )

# %% RBF kernel

rbf_kernel_svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_clf', SVC(kernel='rbf', gamma=5, C=0.001))
])
rbf_kernel_svm_clf.fit( X , y )