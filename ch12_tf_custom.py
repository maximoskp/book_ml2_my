#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 07:22:44 2021

@author: max
"""

import tensorflow as tf

# %% create matrix / scalar

x = tf.constant( [[1.,2.,3.] , [4.,5.,6.] ] )
y = tf.constant( 38 )

print('x: ', repr(x))
print('y: ', repr(y))

print('x.shape: ' + repr(x.shape))
print('x.dtype: ' + repr(x.dtype))

# %% indexing

x1 = x[:, 1:]
print('x1: ' + repr(x1))
x2 = x[..., 1, tf.newaxis] # tf.newaxis: creates shape (2,1) instead of (2,)
print('x2: ' + repr(x2))

# %% operations

x3 = x + 10
x4 = tf.square(x)
x5 = x @ tf.transpose(x) # @ -> tf.matmul

# %% keras also has low lever

from tensorflow import keras

K = keras.backend

y = K.square( K.transpose( x ) ) + 10
print('y: ' + repr(y))

# %% tf - numpy

import numpy as np

a = np.array( [2., 4., 5.] )
t = tf.constant( a )
a1 = t.numpy()

a2 = tf.square( a )
t2 = np.square( t )

# %% type missmatch

# not allowed
# tf.constant(2.) + tf.constant(42)

# not allowed
# tf.constant(2.) + tf.constant(42., dtype=tf.float64)

# allowed
t1 = tf.constant(2.)
t2 = tf.constant(42)
t1 + tf.cast(t2, tf.float32)

# %% variables

v = tf.Variable( [ [1.,2.,3.] , [4.,5.,6.] ] )
print( 'v: ' + repr(v) )

v.assign(2*v)
print( 'v: ' + repr(v) )

v[0,1].assign(42)
print( 'v: ' + repr(v) )

v[:,2].assign([0., 1.])
print( 'v: ' + repr(v) )

v.scatter_nd_update( indices=[[0,0], [1,2]], updates=[100.,200.] )
print( 'v: ' + repr(v) )

# %% custom loss function with custom parameter

# - use only tf functions, not np, for taking full advantage of acceleration
# - check out convienient syntax {**x}

class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error)/2
        linear_loss = self.threshold * tf.abs(error) - self.threshold**2/2
        return tf.where( is_small_error, squared_loss, linear_loss )
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'threshold': self.threshold}

# incorporated in model
# model.compile( loss=HuberLoss(2.), optimizer='nadam' )

# then it can be loaded as
# model = keras.models.load_model('name.h5',
#                                custom_objects={'HuberLoss': HuberLoss()})

# %% autodiff - traditional gradients

def f(w1, w2):
    return 3*w1**2 + 2 + 2*w1*w2

w1, w2 = 5, 3
eps = 1e-6

w1_diff = ( f(w1+eps, w2) - f(w1, w2) )/eps
print( 'w1_diff: ' + str( w1_diff ) )
w2_diff = ( f(w1, w2+eps) - f(w1, w2) )/eps
print( 'w2_diff: ' + str( w2_diff ) )

# %% autodiff - tf approach

w1, w2 = tf.Variable(5.), tf.Variable(3.)

with tf.GradientTape() as tape:
    z = f(w1, w2)

gradients = tape.gradient(z, [w1, w2])
print('gradients: ' + repr(gradients))















