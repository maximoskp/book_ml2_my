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

# %% classification with two layers

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

'''
# alternative initialization
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28])
    keras.layers.Dense(300, activation='relu')
    keras.layers.Dense(100, activation='relu')
    keras.layers.Dense(10, activation='softmax')
])
'''

# %% print summary of the model
model.summary()

# %% see layers
print(repr(model.layers))
# get layer
hidden1 = model.layers[1]
print(hidden1.name)

# %% access weights and biases
weights, biases = hidden1.get_weights()
print(repr(weights.shape))
print(repr(biases.shape))

# %% compile model

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# %% train model

history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))

# %% plot training history

import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot()
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

# validation error seems to be less than training error in each epoch
# but this happens because validation error is computed after the 
# the training within this epoch has been completed, while training 
# error is computed during training.
# Check comment in p.305: training error should be shifted half epoch