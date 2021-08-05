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

# %% resnet-34

class ResidualUnit(keras.layers.Layer):
    def __init__( self, filters, strides=1, activation='relu', **kwargs ):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides,
                                padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, 3, strides=1,
                                padding='same', use_bias=False),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides=strides,
                                    padding='same',use_bias=False),
                keras.layers.BatchNormalization()]
    # end init
    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation( Z + skip_Z )
    # end call
# end ResidualUnit

model = keras.models.Sequential()
model.add( keras.layers.Conv2D(64, 7, strides=2, input_shape=[224,224,3],
                               padding='same', use_bias=False) )
model.add( keras.layers.BatchNormalization() )
model.add( keras.layers.Activation('relu') )
model.add( keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same') )
prev_filters = 64
for filters in [64]*3 + [128]*4 + [256]*6 + [512]*3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters
model.add( keras.layers.GlobalAvgPool2D() )
model.add( keras.layers.Flatten() )
model.add( keras.layers.Dense(10, activation='softmax') )

# %% resize

X_train_res = tf.image.grayscale_to_rgb( tf.image.resize(X_train, [224,224]) )
X_valid_res = tf.image.grayscale_to_rgb( tf.image.resize(X_valid, [224,224]) )

# %% train

history = model.fit( X_train_res, y_train, epochs=30,
                    validation_data=(X_valid_res, y_valid))



