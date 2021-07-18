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

plt.clf()
plt.imshow(weights)
plt.show()

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

# %% evaluate model

model.evaluate( X_test, y_test )

# %% make predictions

X_new = X_test[:3]

y_proba = model.predict(X_new)
print(y_proba.round(2))

# if only interested in prediction label
y_pred = model.predict_classes(X_new)
print(y_pred.round(2))

plt.clf()
plt.subplot(131)
plt.imshow(X_new[0], cmap='gray')
plt.title( 'pred: ' + str(y_pred[0]) + ' - true: ' + str(y_test[0]) )
plt.subplot(132)
plt.imshow(X_new[1], cmap='gray')
plt.title( 'pred: ' + str(y_pred[1]) + ' - true: ' + str(y_test[1]) )
plt.subplot(133)
plt.imshow(X_new[2], cmap='gray')
plt.title( 'pred: ' + str(y_pred[2]) + ' - true: ' + str(y_test[2]) )

# %% regresssion - california housing

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %%

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_valid = scaler.fit_transform(X_valid)
X_test = scaler.fit_transform(X_test)

# %% construct model

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='sgd')

# %% train

history = model.fit( X_train, y_train, epochs=20, validation_data=(X_valid, y_valid) )
# MSE error will not display accuracies

# %% predict

X_new = X_test[:3]
y_pred = model.predict( X_new )

print('predicted: ' + repr(y_pred))
print('true: ' + repr(y_test[:3]))

# %% functional api

input_ = keras.layers.Input(shape=X_train.shape[1:])
# create hidder an directly pass input_ as a function
hidden1 = keras.layers.Dense( 30, activation='relu' )(input_)
hidden2 = keras.layers.Dense( 30, activation='relu' )(hidden1)
concat = keras.layers.Concatenate()([input_, hidden2])
# or
# concat = keras.layers.concatenate( [input_, hidden2] )
output = keras.layers.Dense(1)(concat)

model = keras.Model( inputs=[input_], outputs=[output] )

model.summary()

# %% multiple inputs

inputA = keras.layers.Input(shape=[5], name='wide_input')
inputB = keras.layers.Input(shape=[6], name='deep_input')
hidden1 = keras.layers.Dense(30, activation='relu')(inputB)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
concat = keras.layers.Concatenate()([inputA, hidden2])
output = keras.layers.Dense(1, name='output')(concat)

model = keras.Model(inputs=[inputA, inputB], outputs=[output])

model.summary()

# %% train / test with multiple inputs

X_train_A , X_train_B = X_train[:, :5] , X_train[:, 2:]
X_valid_A , X_valid_B = X_valid[:, :5] , X_valid[:, 2:]
X_test_A , X_test_B = X_test[:, :5] , X_test[:, 2:]

X_new_A , X_new_B = X_test_A[:3] , X_test_B[:3]
y_new = y_test[:3]

model.compile(loss='mse', optimizer=keras.optimizers.SGD(lr=1e-3))

'''
history = model.fit(
    (X_train_A, X_train_B), y_train, epochs=20,
    validation_data=((X_valid_A, X_valid_B), y_valid)
)
'''

# dictionary for inputs to avoid confusion
history = model.fit(
    {'wide_input': X_train_A, 'deep_input': X_train_B}, y_train, epochs=20,
    validation_data=((X_valid_A, X_valid_B), y_valid)
)

# %% evaluate
mse_test = model.evaluate( [X_test_A, X_test_B], y_test )
print('mse_test: ' + repr(mse_test))
y_pred = model.predict( [X_new_A, X_new_B] )
print('y_pred: ' + repr(y_pred))
print('true: ' + repr(y_new))

# %% multiple outputs

inputA = keras.layers.Input(shape=[5], name='wide_input')
inputB = keras.layers.Input(shape=[6], name='deep_input')
hidden1 = keras.layers.Dense(30, activation='relu')(inputB)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
concat = keras.layers.Concatenate()([inputA, hidden2])
output = keras.layers.Dense(1, name='main_output')(concat)
aux_output = keras.layers.Dense(1, name='aux_output')(hidden2)

model = keras.Model(inputs=[inputA, inputB], outputs=[output, aux_output])

model.summary()

# %% cost function of outputs

model.compile( loss=['mse', 'mse'], loss_weights=[0.9, 0.1], optimizer='sgd' )

history = model.fit(
    [X_train_A, X_train_B], [y_train, y_train], epochs=20,
    validation_data=( [X_valid_A, X_valid_B], [y_valid, y_valid] )
)

# %% evaluate
mse_test = model.evaluate( [X_test_A, X_test_B], [y_test, y_test] )
print('mse_test: ' + repr(mse_test))
y_pred = model.predict( (X_new_A, X_new_B) )
print('y_pred: ' + repr(y_pred))
print('true: ' + repr(y_new))

# %% subclassing API

class WideAndDeepModel(keras.Model):
    def __init__(self, units=30, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)
    # end init
    
    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate( [inputA, hidden2] )
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output
    # end call
# end class

model = WideAndDeepModel()

# %%

'''
model.compile(loss='mse', optimizer=keras.optimizers.SGD(lr=1e-3))
model.fit( (inputA, inputB), y_train )
model.summary()
'''

# %% saving / loading a model - not working so simply with subclassing

inputA = keras.layers.Input(shape=[5], name='wide_input')
inputB = keras.layers.Input(shape=[6], name='deep_input')
hidden1 = keras.layers.Dense(30, activation='relu')(inputB)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
concat = keras.layers.Concatenate()([inputA, hidden2])
output = keras.layers.Dense(1, name='main_output')(concat)
aux_output = keras.layers.Dense(1, name='aux_output')(hidden2)

model = keras.Model(inputs=[inputA, inputB], outputs=[output, aux_output])

model.summary()

model.compile( loss=['mse', 'mse'], loss_weights=[0.9, 0.1], optimizer='sgd' )

history = model.fit(
    [X_train_A, X_train_B], [y_train, y_train], epochs=20,
    validation_data=( [X_valid_A, X_valid_B], [y_valid, y_valid] )
)

# %% saving

import os

model.save('models'+os.sep+'ch10_test_model.h5')

# %% loading

model = keras.models.load_model('models'+os.sep+'ch10_test_model.h5')
model.summary()

# %% callbacks

nputA = keras.layers.Input(shape=[5], name='wide_input')
inputB = keras.layers.Input(shape=[6], name='deep_input')
hidden1 = keras.layers.Dense(30, activation='relu')(inputB)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
concat = keras.layers.Concatenate()([inputA, hidden2])
output = keras.layers.Dense(1, name='main_output')(concat)
aux_output = keras.layers.Dense(1, name='aux_output')(hidden2)

model = keras.Model(inputs=[inputA, inputB], outputs=[output, aux_output])

model.summary()

model.compile( loss=['mse', 'mse'], loss_weights=[0.9, 0.1], optimizer='sgd' )

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    'models'+os.sep+'ch10_checkpoint.h5',
    save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    [X_train_A, X_train_B], [y_train, y_train], epochs=20,
    validation_data=( [X_valid_A, X_valid_B], [y_valid, y_valid] ),
    callbacks=[checkpoint_cb, early_stopping_cb]
)

# %% custom callbacks

class PrintRatio(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print( '\nval/train: {:.2f}'.format(logs['val_loss']/logs['loss']) )

ratio_print_cb = PrintRatio()

history = model.fit(
    [X_train_A, X_train_B], [y_train, y_train], epochs=20,
    validation_data=( [X_valid_A, X_valid_B], [y_valid, y_valid] ),
    callbacks=[checkpoint_cb, early_stopping_cb, ratio_print_cb]
)

# %% tensorboard

import os
root_logdir = os.path.join(os.curdir, 'my_logs')

def get_run_logdir():
    import time
    run_id = time.strftime( 'run_%Y_%m_%d-%H_%M_%S' )
    return os.path.join(root_logdir, run_id)
# end getrun_logdir

# %% 

tensorboard_cb = keras.callbacks.TensorBoard( get_run_logdir() )

history = model.fit(
    [X_train_A, X_train_B], [y_train, y_train], epochs=20,
    validation_data=( [X_valid_A, X_valid_B], [y_valid, y_valid] ),
    callbacks=[tensorboard_cb]
)

# %% create custom tensorflow logging

test_logdir = get_run_logdir()

writer = tf.summary.create_file_writer( test_logdir )

with writer.as_default():
    for step in range(1, 1000 + 1):
        tf.summary.scalar('my_scalar', np.sin(step/10), step=step)
        
        data = (np.random.randn(100) + 2)*step/100
        tf.summary.histogram('my_hist', data, buckets=50, step=step)
        
        images = np.random.rand(2, 32, 32, 3)
        tf.summary.image('my_images', images*step/1000, step=step)
        
        texts = ['The step is ' + str(step) + ' - squared:' + str(step**2)]
        tf.summary.text('my_tesxt', texts, step=step)
        
        sine_wave = tf.math.sin(tf.range(12000)/44100*2*np.pi*step)
        audio = tf.reshape(tf.cast(sine_wave, tf.float32), [1, -1, 1])
        tf.summary.audio('my_audio', audio, sample_rate=44100, step=step)
        

# %% fine-tuning hyperparameters

def build_model( n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8] ):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation='relu'))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model

# %% can be used as a standard sklearn regressor

keras_reg = keras.wrappers.scikit_learn.KerasRegressor( build_model )

keras_reg.fit(
    X_train, y_train, epochs=100,
    validation_data=(X_valid, y_valid),
    callbacks=[keras.callbacks.EarlyStopping(patience=10)]
)

# %%

mse_test = keras_reg.score( X_test, y_test )
print( 'mse_test: ' +  repr(mse_test) )
y_pred = keras_reg.predict(X_new)
print( 'y_pred: ' +  repr(y_pred) )

# %% randomized search

# not working - need to downgrade scikit
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    'n_hidden': [0, 1, 2, 3],
    'n_neurons': np.arange(1, 100),
    'learning_rate': reciprocal(3e-4, 3e-2)
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(
    X_train, y_train, epochs=100,
    validation_data=(X_valid, y_valid),
    callbacks=[keras.callbacks.EarlyStopping(patience=10)]
)

# %%

print(rnd_search_cv.best_params_)
print(rnd_search_cv.best_score_)

model = rnd_search_cv.best_estimator_.model







# ------