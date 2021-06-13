#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 23:47:54 2021

@author: max
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

blob_centers = np.array([
     [0.2,  2.3],
     [-1.5 ,  2.3],
     [-2.8,  1.8],
     [-2.8,  2.8],
     [-2.8,  1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

X, y = make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std, random_state=7)

plt.clf()
plt.scatter( X[:, 0], X[:, 1], c=y, s=1 )
plt.show()

# %% k-means

from sklearn.cluster import KMeans

k = 5
kmeans = KMeans(n_clusters=k)
y_pred = kmeans.fit_predict(X)

# centroids
c = kmeans.cluster_centers_

plt.clf()
plt.scatter( X[:, 0], X[:, 1], c=y_pred, s=1 )
plt.plot( c[:, 0] , c[:, 1], 'rx')
plt.show()

# data that kmeans has been trained on
print('kmeans.labels_: ' + repr(kmeans.labels_))

# %% predict cluster of new instances

X_new = np.array( [ [0,2], [3,2], [-3,3], [-3,2.5] ] )
y_pred_new = kmeans.predict( X_new )
print('y_pred_new: ' + repr(y_pred_new))

# %% transform data on distances from clusters

t = kmeans.transform( X_new )
print('transformed: ' + repr(t))

# %% initization

# define initial centroids
good_init = np.array([ [-3,3], [-3,2], [-3,1], [-1,2], [0,2] ])
kmeans = KMeans(n_clusters=5, init=good_init, n_init=1)

# By default, when fit is called, it runs random initialisations 10 times
# (defined by an int in n_init) and keeps the best solution.
# Inertia is employed for solution evaluation, which measures the mean squared
# distance between each instance and its centroid

kmeans.fit( X )
print('inertia: ' + str(kmeans.inertia_) )

# Score shows the negative inertia - greater is better
print('score: ' + str(kmeans.score(X)) )

# If init is not set to 'random', then clusters are initilised based on the
# kmeans++ algorithm, which favours placing initial centroids remotely 
# from each other.

# %% minibatch kmeans

# accelerates kmeans by using random subsets of instances

from sklearn.cluster import MiniBatchKMeans

minibatch_kmeans = MiniBatchKMeans(n_clusters=5)
minibatch_kmeans.fit( X )
# faster but with slightly worse inertia
print('inertia: ' + str(minibatch_kmeans.inertia_) )

# %% defining the number of clusters

# elbow-like shape is a bit vague
# silhouette score: (extra-intra)/max(extra,intra)
from sklearn.metrics import silhouette_score

s = []
ks = range(2, 9, 1)
kmeans_all = []
for k in ks:
    kmeans = KMeans( n_clusters=k )
    kmeans.fit( X )
    s.append( silhouette_score( X , kmeans.labels_ ) )
    kmeans_all.append( kmeans )

print('silhouette scores: ' + repr(s))
plt.clf()
plt.plot( ks, s, '.--' )
plt.show()

# %% 

from sklearn.metrics import silhouette_samples
silhouette_s = silhouette_samples( X , kmeans_all[3].labels_ )
y_pred = kmeans_all[3].fit_predict( X )

# for each cluster
coeffs = []
for k in range(5):
    coeffs.append( silhouette_s[y_pred == k] )

plt.clf()
plt.boxplot( coeffs )
plt.show()

# %% image segmentation

import os
from matplotlib.image import imread
image = imread( os.path.join( 'images', 'unsupervised_learning', 'ladybug.png' ) )

print( repr( image.shape ) )

# %% 

X = image.reshape(-1, 3)

# we need above 8 clusters for having the red color on the bug
kmeans = KMeans( n_clusters=4 ).fit(X)

segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape( image.shape )

plt.clf()
plt.imshow( segmented_img )
plt.show()

# %% preprocessing mnist

from sklearn.datasets import load_digits
X_digits, y_digits = load_digits( return_X_y=True )

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X_digits, y_digits )

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit( X_train, y_train )

no_preprocess = log_reg.score( X_test, y_test )
print( 'no_preprocess score: ' + str(no_preprocess) )

# %% 

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('kmeans', KMeans(n_clusters=50)),
    ('log_reg', LogisticRegression()) ])
pipeline.fit( X_train , y_train )

preprocess_score = pipeline.score( X_test, y_test )
print( 'preprocess score: ' + str(preprocess_score) )

# %% grid search for finding proper number of clusters

from sklearn.model_selection import GridSearchCV

param_grid = dict( kmeans__n_clusters=range(2,100) )

grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
grid_clf.fit(X_train, y_train)

best_params = grid_clf.best_params_
print('best_params: ' + repr(best_params))
grid_score = grid_clf.score( X_test, y_test )
print('grid_score: ' + repr(grid_score))

# %% semi-supervised learning

# keep a small part of data
n_labeled = 50

log_reg = LogisticRegression()
log_reg.fit( X_train[:n_labeled], y_train[:n_labeled] )

first_score = log_reg.score( X_test, y_test )
print('first_score: ' + str(first_score))

# %% get 50 representative images instead of 50 random

k = 50
kmeans = KMeans( n_clusters=k )
X_digits_dist = kmeans.fit_transform( X_train )
representative_digit_idx = np.argmin( X_digits_dist , axis=0 )
X_representative_digit = X_train[representative_digit_idx]
# annotate manually
y_representative_digits = []
for i in range(k):
    plt.clf()
    plt.imshow( X_representative_digit[i,:].reshape(8,8) )
    plt.pause(0.1)
    n = input('asign label: ')
    y_representative_digits.append( int(n) )

# %%

log_reg = LogisticRegression()
log_reg.fit( X_representative_digit, y_representative_digits )
second_score = log_reg.score( X_test, y_test )
print('second_score: ' + str(second_score))

# %% label propagation

# assign manual label to all cluster members
y_train_propagated = np.empty( len(X_train) , dtype=np.int32 )

for i in range(k):
    y_train_propagated[ kmeans.labels_ == i ] = y_representative_digits[i]

log_reg = LogisticRegression()
log_reg.fit( X_train, y_train_propagated )
third_score = log_reg.score( X_test, y_test )
print('third_score: ' + str(third_score))

# %% propagate only 20% of the instances closer to the centroid

percentile_closest = 20

X_cluster_dist = X_digits_dist[ np.arange( len(X_train) ) , kmeans.labels_ ]
for i in range(k):
    in_cluster = (kmeans.labels_ == i)
    cluster_dist = X_cluster_dist[ in_cluster ]
    cutoff_distance = np.percentile( cluster_dist, percentile_closest )
    above_cutoff = ( X_cluster_dist > cutoff_distance )
    X_cluster_dist[ in_cluster & above_cutoff ] = -1

partially_propagated = ( X_cluster_dist != -1 )

X_train_partially_propagated = X_train[ partially_propagated ]
y_train_partially_propagated = y_train[ partially_propagated ]

# %% train with partially propagated

log_reg = LogisticRegression()
log_reg.fit( X_train_partially_propagated, y_train_partially_propagated )
fourth_score = log_reg.score( X_test, y_test )
print('fourth_score: ' + str(fourth_score))

# partially propagated labeled correctly
correctly_labeled = np.mean( y_train_partially_propagated == y_train[ partially_propagated ] )
print('correctly_labeled: ' + str(correctly_labeled))

# %% dbscan

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=1000, noise=0.05)
dbscan = DBSCAN( eps=0.05, min_samples=5 )
dbscan.fit( X )

# %%

plt.clf()
# plt.scatter( X[:,0], X[:,1], c=dbscan.labels_ )
plt.scatter( dbscan.components_[:,0], dbscan.components_[:,1], c=dbscan.labels_[ dbscan.core_sample_indices_ ] )
# anomalies (labelled as -1)
plt.plot( X[ dbscan.labels_ == -1 ,0], X[ dbscan.labels_ == -1 ,1], 'xr' )
plt.show()

# %% larger epsilon

dbscan = DBSCAN( eps=0.2, min_samples=5 )
dbscan.fit( X )

plt.clf()
# plt.scatter( X[:,0], X[:,1], c=dbscan.labels_ )
plt.scatter( dbscan.components_[:,0], dbscan.components_[:,1], c=dbscan.labels_[ dbscan.core_sample_indices_ ] )
# anomalies (labelled as -1)
plt.plot( X[ dbscan.labels_ == -1 ,0], X[ dbscan.labels_ == -1 ,1], 'xr' )
plt.show()

# %% DBSCAN does not include a predict function
# new instances can be assigned to a cluster by knn

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier( n_neighbors=50 )
knn.fit( dbscan.components_ , dbscan.labels_[ dbscan.core_sample_indices_ ] )

# %% new instances

X_new = np.array( [ [-.5,0] , [0,.5], [1,-.1], [2,1] ] )
knn_pred = knn.predict(X_new)
print( 'knn_pred: ' + str(knn_pred) )
knn_proba = knn.predict_proba(X_new)
print( 'knn_proba: ' + repr(knn_proba) )

# %% identify if new instances are outliers

y_dist, y_pred_idx = knn.kneighbors( X_new , n_neighbors=1 )

y_pred = dbscan.labels_[ dbscan.core_sample_indices_ ][ y_pred_idx ]
y_pred[ y_dist > 0.2 ] = -1
print( 'cluster attribution: ' + repr( y_pred.ravel() ) )

# %% GMM

X1, y1 = make_blobs(n_samples=1000, centers=((4,-4),(0,0)), random_state=42)
X1 = X1.dot( np.array([ [0.374, 0.95] , [0.732, 0.598] ]) )
X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
X2 = X2 + [6,-8]

X = np.r_[X1, X2]
y = np.r_[y1, y2+2]

plt.clf()
plt.scatter( X[:,0], X[:,1], c=y )
plt.show()

# %% 

from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=3, n_init=10)
gm.fit(X)

print('weights: ' + repr(gm.weights_))
print('means: ' + repr(gm.means_))
print('covariances: ' + repr(gm.covariances_))
print('converged: ' + str(gm.converged_))
print('iterations: ' + str(gm.n_iter_))

# %% predicting 

X_new = np.array( [ [-3,1], [2,-2] ] )
y_pred = gm.predict(X_new)
y_pred_proba = gm.predict_proba(X_new)
print('y_pred: ' + repr(y_pred))
print('y_pred_proba: ' + repr(y_pred_proba))

# %% generating

X_new, y_new = gm.sample(6)
print('X_new: ' + repr(X_new))
print('y_new: ' + repr(y_new))
# get the log pdf of locations
score_samples = gm.score_samples(X_new)
print('score_samples: ' + repr(score_samples))

# %% anomaly detection

# as a percentile of the lowest assigned probabilities
densities = gm.score_samples(X)
density_threshold = np.percentile( densities , 4 )
anomalies = X[ densities < density_threshold ]

plt.clf()
plt.scatter( X[:,0], X[:,1], c=y )
plt.plot( anomalies[:,0], anomalies[:,1], 'xr' )
plt.show()

# %% number of clusters

bic = []
aic = []

for k in range(2, 6):
    gm = GaussianMixture(n_components=k, n_init=10)
    gm.fit(X)
    bic.append( gm.bic(X) )
    aic.append( gm.aic(X) )

plt.clf()
plt.plot( range(2,6), bic, 'b--.', label='bic' )
plt.plot( range(2,6), aic, 'g--.', label='aic' )
plt.legend()
plt.show()

# %% Bayesian GMMs

from sklearn.mixture import BayesianGaussianMixture

bgm = BayesianGaussianMixture( n_components=10 , n_init=10 )
bgm.fit(X)
print('weights: ' + repr( np.round(bgm.weights_ , 2)))