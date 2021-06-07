#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 08:08:44 2021

@author: max
"""

# %% create dataset

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

m = 600
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m)*3*np.pi/2 - 0.5
X = np.empty( (m, 3) )
X[:, 0] = np.cos( angles ) + np.sin( angles )/2 + noise*np.random.randn(m)/2
X[:, 1] = np.sin( angles )*0.7 + noise*np.random.randn(m)
X[:, 2] = X[:, 0]*w1 + X[:, 1]*w2 + noise*np.random.randn(m)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot( X[:, 0], X[:, 1], X[:, 2], '.' )
plt.show()

# %% center data and apply SVD

X_centered = X - X.mean(axis=0)

U, s, Vt = np.linalg.svd( X_centered )
# https://gregorygundersen.com/blog/2018/12/10/svd/
# http://www.ams.org/publicoutreach/feature-column/fcarc-svd
# principal components
c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]

# %% project in 2d

W2 = Vt.T[:, :2]
X2D = X_centered.dot(W2)

plt.clf()
plt.plot(X2D[:,0], X2D[:, 1], '.')
plt.show()

# %% PCA by scikit-learn

from sklearn.decomposition import PCA

pca = PCA( n_components=2 )
X2D = pca.fit(X)

# get principal components
W2 = pca.components_.T
# get explained variance
v = pca.explained_variance_ratio_
print('explained variance ratio: ', repr(v))

# %% import and fix mnist

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.target = mnist.target.astype(np.uint8)

# %% explained variance, handy functions

from sklearn.model_selection import train_test_split

X = mnist['data']
y = mnist['target']

X_train, X_test, y_train, y_test = train_test_split(X, y)

# %% examine desired explained ratio level

pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum( pca.explained_variance_ratio_ )
d = np.argmax(cumsum > 0.95) + 1
print('dimensions to keep: ' + str(d))

# %% built-in option to preserve given explained variance ratio

pca = PCA( n_components=0.95 )
X_reduced = pca.fit(X_train)

# %% pca for compression

pca = PCA(n_components=154)
X_reduced = pca.fit_transform(X_train)
X_recovered = pca.inverse_transform( X_reduced )

# %%

def plot_two_digits( d1 , d2 ):
    sd1 = d1.reshape(28,28)
    sd2 = d2.reshape(28,28)
    plt.subplot(121)
    plt.imshow( sd1 , cmap='binary' )
    plt.axis( 'off' )
    plt.subplot(122)
    plt.imshow( sd2 , cmap='binary' )
    plt.axis( 'off' )
    plt.show()

for i in range(10):
    plt.clf()
    idx = np.random.randint(1,high=X_train.shape[0])
    plot_two_digits( X_train[idx, :] , X_recovered[idx, :] )
    plt.pause(1.)

# %% randomised PCA

rnd_pca = PCA(n_components=2, svd_solver='randomized') # or full
# in heavy compression with many data, default 'auto' is randomized
X_reduced = rnd_pca.fit_transform(X_train)

# %% incremental pca

from sklearn.decomposition import IncrementalPCA

n_batches = 100

inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split( X_train, n_batches ):
    inc_pca.partial_fit( X_batch )

X_reduced = inc_pca.transform( X_train )

# %% loading data from ROM when needed

# first, save in ROM to make it work
filename = 'my_mnist.data'
m,n = X_train.shape
X_mm = np.memmap( filename, dtype='float32', mode='write', shape=(m,n) )
X_mm[:] = X_train
del X_mm # triggers finalizer, ensuring it's writen to dist

# %% now actually load it from ROM

X_mm = np.memmap( filename, dtype='float32', mode='readonly', shape=(m,n) )

batch_size = m // n_batches
inc_pca = IncrementalPCA(n_components=2, batch_size=batch_size)

inc_pca.fit( X_mm )

# %% kernel pca

# make swiss roll dataset
from sklearn.datasets import make_swiss_roll

X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

from sklearn.decomposition import KernelPCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

rbf_pca = KernelPCA( n_components=2, kernel='rbf', gamma=0.04, fit_inverse_transform=True )
X_rbf = rbf_pca.fit_transform(X)

lin_pca = KernelPCA( n_components=2, kernel='linear', fit_inverse_transform=True )
X_lin = lin_pca.fit_transform(X)

sig_pca = KernelPCA( n_components=2, kernel='sigmoid', gamma=0.01, coef0=1, fit_inverse_transform=True )
X_sig = sig_pca.fit_transform(X)

plt.clf()
plt.subplot(221)
plt.scatter(X[:,0], X[:,1], c=t, cmap=plt.cm.hot)
plt.subplot(222)
plt.scatter(X_rbf[:,0], X_rbf[:,1], c=t, cmap=plt.cm.hot)
plt.subplot(223)
plt.scatter(X_lin[:,0], X_lin[:,1], c=t, cmap=plt.cm.hot)
plt.subplot(224)
plt.scatter(X_sig[:,0], X_sig[:,1], c=t, cmap=plt.cm.hot)
plt.show()

# %% kernel parameters search

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

clf = Pipeline( [
        ( 'kpca', KernelPCA(n_components=2) ),
        ( 'log_reg', LogisticRegression() )
    ] )

param_grid = [{
        'kpca__gamma': np.linspace(0.03, 0.05, 10),
        'kpca__kernel': ['rbf', 'sigmoid']
    }]

grid_search = GridSearchCV(clf, param_grid, cv=3)

y = t > 6.9
grid_search.fit(X, y)

print(grid_search.best_params_)

# %% reconstruction error in pca

# need ot set fit_inverse_transform=True for having available
# inverse_transform function in kernel pca.

rbf_pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.0433,
                    fit_inverse_transform=True)
X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform( X_reduced )

from sklearn.metrics import mean_squared_error
mse = mean_squared_error( X_preimage , X )
print('mse: ' + str(mse))

# %% manifold learning

# LLE
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_reduced = lle.fit_transform(X)

plt.clf()
plt.scatter( X_reduced[:,0], X_reduced[:, 1], c=t, cmap=plt.cm.hot )