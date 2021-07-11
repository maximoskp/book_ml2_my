# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

# %% perceptron

iris = load_iris()
X = iris.data[:, (2, 3)]
y = (iris.target == 0).astype(int)

per_clf = Perceptron()
per_clf.fit(X, y)

y_pred = per_clf.predict( [[2, 0.5]] )

# %%  fashion MNIST

import tensorflow as tf
from tensorflow import keras

# %% 

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# %% check out data

import matplotlib.pyplot as plt

# %%

print('shape: ' + repr(X_train_full.shape) , ' - type: ' + str(X_train_full.dtype))
tmp_idx = np.random.randint(X_train_full.shape[0])
plt.clf()
plt.imshow(X_train_full[tmp_idx,:,:], cmap='gray')
plt.show()

# %% make validation set
# also readjust range to float 0-1 for gradient descend training

X_valid, X_train = X_train_full[:5000]/250., X_train_full[5000:]/250.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test/250.

class_names = ['T-shirt-top', 'Trousers', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']