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

def my_preprocess(image):
    return tf.image.grayscale_to_rgb( tf.image.resize(image, [224,224]) )

batch_size = 32
X_train_res = X_train.map( my_preprocess ).batch( batch_size ).prefetch( 1 )
X_valid_res = X_valid.map( my_preprocess ).batch( batch_size ).prefetch( 1 )
# NOT working - X_train is np array

# %% train

history = model.fit( X_train_res, y_train, epochs=30,
                    validation_data=(X_valid_res, y_valid))

# %% pretrained

model = keras.applications.resnet50.ResNet50(weights='imagenet')

# %% resize example

images_resized = tf.image.resize( images, [224, 224] )
# for preserving aspect ratio:
# images_resized = tf.image.crop_and_resize( images, [224, 224] )

# %%

plt.subplot(121)
plt.imshow( images_resized[0] )
plt.subplot(122)
plt.imshow( images_resized[1] )

# %% preprocess input

inputs = keras.applications.resnet50.preprocess_input(images_resized*255)

# %% prediction

Y_proba = model.predict( inputs )

# %% 

top_5 = keras.applications.resnet50.decode_predictions(Y_proba, top=5)

for img_idx in range( len( images_resized ) ):
    print( 'image: ' + str(img_idx) )
    for class_id, name, y_proba in top_5[ img_idx ]:
        print( '{} - {:12s} {:.2f}%'.format( class_id, name, y_proba*100 ) )

# %% transfer learning - load data

import tensorflow_datasets as tfds

dataset, info = tfds.load( 'tf_flowers', as_supervised=True, with_info=True )

# %% explore info

dataset_size = info.splits['train'].num_examples
class_names = info.features['label'].names
n_classes = info.features['label'].num_classes

# %% get with train-validation-test

test_set_raw, valid_set_raw, train_set_raw = tfds.load(
    'tf_flowers',
    split=['train[:10%]', 'train[10%:25%]', 'train[25%:]'],
    as_supervised=True)

# %% 

def preprocess(image, label):
    resized_image = tf.image.resize( image, [224, 224] )
    final_image = keras.applications.xception.preprocess_input( resized_image )
    return final_image, label

batch_size = 12
train_set = train_set_raw.shuffle(1000)
train_set = train_set.map( preprocess ).batch( batch_size ).prefetch( 1 )
valid_set = valid_set_raw.map( preprocess ).batch( batch_size ).prefetch( 1 )
test_set = test_set_raw.map( preprocess ).batch( batch_size ).prefetch( 1 )


# %% get pretrained model without top

base_model = keras.applications.xception.Xception(weights='imagenet',
                                                  include_top=False)

avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense( n_classes, activation='softmax' )(avg)
model = keras.Model(inputs=base_model.input, outputs=output)

# %% 

for layer in base_model.layers:
    layer.learnable = False


# %% 
optimizer = keras.optimizers.SGD( lr=0.2, momentum=0.9, decay=0.01 )
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])
history = model.fit(train_set, epochs=5, validation_data=valid_set)

# %% unfreeze no

for layer in base_model.layers:
    layer.trainable=True

optimizer = keras.optimizers.SGD( lr=0.01, momentum=0.9, decay=0.001 )
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])
history = model.fit(train_set, epochs=5, validation_data=valid_set)





