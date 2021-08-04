#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 07:48:39 2021

@author: max
"""

from sklearn.datasets import load_sample_image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

china = load_sample_image('china.jpg')/255
flower = load_sample_image('flower.jpg')/255

images = np.array( [china , flower] )

plt.subplot(121)
plt.imshow(china)
plt.subplot(122)
plt.imshow(flower)

# %% 

batch_size, height, width, channels = images.shape

# create 2 filters
filters = np.zeros( shape=(27, 27, channels, 2) , dtype=np.float32 )
# vertical and horizontal line
filters[ :, 13, :, 0 ] = 1
filters[ 13, :, :, 1 ] = 1

plt.subplot(121)
plt.imshow(filters[:,:,:,0], cmap='gray')
plt.subplot(122)
plt.imshow(filters[:,:,:,1], cmap='gray')

# %% apply filters

outputs = tf.nn.conv2d( images , filters , strides=1 , padding='SAME' )
plt.subplot(121)
plt.imshow(outputs[0,:,:,0], cmap='gray')
plt.subplot(122)
plt.imshow(outputs[0,:,:,1], cmap='gray')
# plt.savefig('test1.png', dpi=300)

# %% trainable

conv = keras.layers.Conv2D( filters=32, kernel_size=3, strides=1,
                           padding='SAME', activation='relu')
# max pooling
max_pool = keras.layers.MaxPool2D(pool_size=2)
# or AvgPool2D, for mean pooling

# %% example cnn

model = keras.models.Sequential([
    keras.layers.Conv2D(64, 7, activation='relu', padding='SAME',
                        input_shape=[28,28,1]),
    keras.layers.MaxPool2D(2),
    keras.layers.Conv2D(128, 3, activation='relu', padding='SAME'),
    keras.layers.Conv2D(128, 3, activation='relu', padding='SAME'),
    keras.layers.MaxPool2D(2),
    keras.layers.Conv2D(256, 3, activation='relu', padding='SAME'),
    keras.layers.Conv2D(256, 3, activation='relu', padding='SAME'),
    keras.layers.MaxPool2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

model.summary()

model.compile( optimizer='adam', loss='sparse_categorical_crossentropy' )

# %% data

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_valid, X_train = X_train_full[:5000]/250., X_train_full[5000:]/250.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test/250.

X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True) + 1e-7
X_train = (X_train - X_mean) / X_std
X_valid = (X_valid - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]

class_names = ['T-shirt-top', 'Trousers', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# %% train

history = model.fit( X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))
