# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 08:49:36 2021

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_time_series( batch_size, n_steps ):
    freq1, freq2, offset1, offset2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5*np.sin( (time-offset1)*(freq1*10 + 10 ) )
    series += 0.2*np.sin( (time-offset2)*(freq2*20 + 20 ) )
    series += 0.1*( np.random.rand( batch_size, n_steps ) - 0.5 )
    return series[..., np.newaxis].astype(  np.float32 )

# %% 

n_steps = 50
series = generate_time_series( 10000 , n_steps + 1)
X_train, y_train = series[:7000, :n_steps] , series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps] , series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps] , series[9000:, -1]

plt.subplot(2,2,1)
plt.plot( series[ np.random.randint(10000), :,: ], '.-' )
plt.subplot(2,2,2)
plt.plot( series[ np.random.randint(10000), :,: ], '.-' )
plt.subplot(2,2,3)
plt.plot( series[ np.random.randint(10000), :,: ], '.-' )
plt.subplot(2,2,4)
plt.plot( series[ np.random.randint(10000), :,: ], '.-' )

# %% metrics

import tensorflow as tf
from tensorflow import keras

# predict previous value
y_pred =  X_valid[:, -1]
er = np.mean( keras.losses.mean_squared_error( y_valid, y_pred ) )
print('er: ', er)

# %% simple feedforward

model = keras.models.Sequential([
    keras.layers.Flatten( input_shape=[50, 1] ),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit( X_train, y_train, epochs=20 )

y_pred = model.predict( X_valid )
er = np.mean( keras.losses.mean_squared_error( y_valid, y_pred ) )
print('er: ', er)

# %% simple rnn

model = keras.models.Sequential([
    keras.layers.SimpleRNN(1, input_shape=[None, 1]),
])

model.compile(optimizer='adam', loss='mse')

model.fit( X_train, y_train, epochs=20 )

y_pred = model.predict( X_valid )
er = np.mean( keras.losses.mean_squared_error( y_valid, y_pred ) )
print('er: ', er)

# %% 

model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.SimpleRNN(1, return_sequences=True, input_shape=[None, 1]),
])

model.compile(optimizer='adam', loss='mse')

model.fit( X_train, y_train, epochs=20 )

y_pred = model.predict( X_valid )
er = np.mean( keras.losses.mean_squared_error( y_valid, y_pred ) )
print('er: ', er)

# %% 

model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(1),
])

model.compile(optimizer='adam', loss='mse')

model.fit( X_train, y_train, epochs=5 )

y_pred = model.predict( X_valid )
er = np.mean( keras.losses.mean_squared_error( y_valid, y_pred ) )
print('er: ', er)

# %% predicting 10 steps further


series = generate_time_series(1, n_steps + 10)
X_new, Y_new = series[:, :n_steps], series[:, n_steps:]
X = X_new
for step_ahead in range(10):
    y_pred_one = model.predict(X[:, step_ahead:])[:, np.newaxis, :]
    X = np.concatenate([X, y_pred_one], axis=1)

Y_pred = X[:, n_steps:]

plt.plot( series[0], '.-' )
plt.plot( 50+np.arange(10), Y_pred[0], 'rx-' )

# %% seqence to vector

series = generate_time_series( 10000 , n_steps + 10)
X_train, Y_train = series[:7000, :n_steps] , series[:7000, -10:, 0]
X_valid, Y_valid = series[7000:9000, :n_steps] , series[7000:9000, -10, 0]
X_test, Y_test = series[9000:, :n_steps] , series[9000:, -10, 0]

model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(10),
])

model.compile(optimizer='adam', loss='mse')

# %% 

model.fit( X_train, y_train, epochs=5 )

# %% 

y_pred = model.predict( X_new )
er = np.mean( keras.losses.mean_squared_error( y_valid, y_pred ) )
print('er: ', er)

# %% seq to seq

Y = np.empty((10000, n_steps, 10))
for step_ahead in range(1, 10+1):
    Y[:,:, step_ahead-1] = series[:, step_ahead:step_ahead+n_steps, 0]
Y_train = Y[:7000]
Y_valid = Y[7000:9000]
Y_test = Y[9000:]

# %% 

model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None,1]),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.TimeDistributed( keras.layers.Dense(10) )
])

# %%

def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

optimizer = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='mse', optimizer=optimizer, metrics=[last_time_step_mse])

# %% 

model.fit(X_train, Y_train)

# %%

class LNSimpleRNNCell( keras.layers.Layer ):
    def __init__(self, units, activation='tanh', **kwargs):
        super().__init__(**kwargs)
        self.state_size = units
        self.output_size = units
        self.simple_rnn_cell = keras.layers.SimpleRNNCell(units,
                                                          activation=None)
        self.layer_norm = keras.layers.LayerNormalization()
        self.activation = keras.activations.get(activation)
    
    def call(self, inputs, states):
        outputs, new_states = self.simple_rnn_cell(inputs, states)
        norm_outputs = self.activation(self.layer_norm(outputs))
        return norm_outputs, [norm_outputs]

# %% 

model = keras.models.Sequential([
    keras.layers.RNN(LNSimpleRNNCell(20), return_sequences=True,
                     input_shape=[None, 1]),
    keras.layers.RNN(LNSimpleRNNCell(20), return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])

# %% LSTM

model = keras.models.Sequential([
    keras.layers.LSTM(20, return_sequences=True, input_shape=[None,1]),
    keras.layers.LSTM(20, return_sequences=True),
    keras.layers.TimeDistributed( keras.layers.Dense(10) )
])

# %%

def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

optimizer = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='mse', optimizer=optimizer, metrics=[last_time_step_mse])

# %% 

model.fit(X_train, Y_train, epochs=20)

# %% GRU

model = keras.models.Sequential([
    keras.layers.GRU(20, return_sequences=True, input_shape=[None,1]),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.TimeDistributed( keras.layers.Dense(10) )
])


# %%

def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

optimizer = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='mse', optimizer=optimizer, metrics=[last_time_step_mse])

# %% 

model.fit(X_train, Y_train, epochs=20)


# %% conv1d

model = keras.models.Sequential([
    keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, 
                        padding='valid', input_shape=[None, 1]),
    keras.layers.GRU(20, return_sequences=True, input_shape=[None,1]),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.TimeDistributed( keras.layers.Dense(10) )
])

# %%

def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

optimizer = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='mse', optimizer=optimizer, metrics=[last_time_step_mse])

# %% 

history = model.fit(X_train, Y_train[:, 3::2], epochs=20,
                    validation_data=(X_valid, Y_valid[:, 3::2]))

# %% wavenet

model = keras.models.Sequential()
model.add( keras.layers.InputLayer(input_shape=[None, 1]) )
for rate in (1, 2, 4, 8)*2:
    model.add( keras.layers.Conv1D( filters=20, kernel_size=2, padding='causal',
                                   activation='relu', dilation_rate=rate) )
model.add( keras.layers.Conv1D( filters=10, kernel_size=1 ) )
model.compile( loss='mse', optimizer='adam', metrics=[last_time_step_mse] )

# %% 

history = model.fit( X_train, Y_train, epochs=20, 
                    validation_data=(X_valid, Y_valid) )

# %% tests

x = np.arange(10, dtype=np.float32)
# l = keras.layers.SimpleRNN(5, input_shape=[None,1], return_sequences=True)
l = keras.layers.Conv1D( filters=20, kernel_size=3, strides=1,
                        input_shape=[None,1], padding='causal', 
                        dilation_rate=2, kernel_initializer=keras.initializers.Ones,
                        activation=keras.activations.linear )

x = x[ np.newaxis , ... , np.newaxis ]

y = l(x)

z = y.numpy()
w0 = l.weights[0].numpy()
w1 = l.weights[1].numpy()
# plt.imshow(z)
