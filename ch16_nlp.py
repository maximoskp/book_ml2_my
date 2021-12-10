# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 07:53:18 2021

@author: user
"""

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

import os

# %% 

shakespeare_url = 'https://homl.info/shakespeare'
filepath = keras.utils.get_file(os.getcwd()+'/datasets/'+'shakespeare.txt', shakespeare_url)
with open( filepath ) as f:
    shakespeare_text = f.read()

# %% 

tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(shakespeare_text)

# %% stats

print('token indexes: ' + repr(tokenizer.word_index))
c = tokenizer.word_counts
print('token counts: ' + repr(c))
plt.bar( c.keys(), c.values() )

# %% 

s = tokenizer.texts_to_sequences(['Lala lolo'])
print(s)
# starting from 0
s0 = np.array(s) - 1
print(s0)
t = tokenizer.sequences_to_texts([[11,35,2,12]])
print(t)

# %% dataset

max_id = len( tokenizer.word_index )
dataset_size = tokenizer.document_count
train_size = dataset_size*90//100
# train_size = 1000

[encoded] = np.array( tokenizer.texts_to_sequences( [ shakespeare_text ] ) ) - 1
dataset = tf.data.Dataset.from_tensor_slices( encoded[:train_size] )


# %% chop

n_steps = 100

window_length = n_steps + 1
dataset = dataset.window(window_length, shift=1, drop_remainder=True)



# %% flatten

dataset = dataset.flat_map( lambda window: window.batch(window_length) )


# %%

batch_size = 32
dataset = dataset.shuffle(10000).batch(batch_size)
dataset = dataset.map( lambda windows: (windows[:, :-1] , windows[:, 1:]) )

# %%

dataset = dataset.map(
    lambda X_batch, Y_batch: ( tf.one_hot( X_batch, depth=max_id ), Y_batch ))

# %% 

dataset = dataset.prefetch(1)

# %% 

model = keras.Sequential([
    keras.layers.GRU( 128, return_sequences=True, input_shape=[None, max_id] ),
    keras.layers.GRU( 128, return_sequences=True ),
    keras.layers.TimeDistributed(keras.layers.Dense(max_id, activation='softmax'))
])
'''
model = keras.Sequential([
    keras.layers.GRU( 128, return_sequences=True, input_shape=[None, max_id],
                     dropout=0.2, recurrent_dropout=0.2 ),
    keras.layers.GRU( 128, return_sequences=True,
                     dropout=0.2, recurrent_dropout=0.2 ),
    keras.layers.TimeDistributed(keras.layers.Dense(max_id, activation='softmax'))
])

# no GPU because recurrent_dropouts != 0 ???
'''
# %%

model.compile( loss='sparse_categorical_crossentropy', optimizer='adam' )

# %% 

history = model.fit( dataset, epochs=10 )

# %% 

# model.save( 'models/ch16ShakespreareRNN.h5' )
model = keras.models.load_model('models/ch16ShakespreareRNN.h5')

# %% 

def preprocess( texts ):
    X = np.array( tokenizer.texts_to_sequences(texts) ) - 1
    return tf.one_hot( X, max_id )

# %%

X_new = preprocess( ['How are yo'] )

# Y_pred = model.predict_classes(X_new)
Y_pred = np.argmax( model.predict(X_new), axis=-1 )

print( tokenizer.sequences_to_texts( Y_pred + 1 ) )
print( tokenizer.sequences_to_texts( Y_pred + 1 )[0][-1] )

# %%

def next_char( text, temperature=1 ):
    X_new = preprocess([text])
    y_proba = model.predict(X_new)[0, -1:, :]
    rescaled_logits = tf.math.log(y_proba)/temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]

def complete_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text

# %% 

t02 = complete_text('t', temperature=0.2)
print('t02: ' + t02)
t1 = complete_text('t', temperature=1.)
print('t1: ' + t1)

# %% stateful

dataset = tf.data.Dataset.from_tensor_slices( encoded[:train_size] )
dataset = dataset.window(window_length, shift=n_steps, drop_remainder=True)
dataset = dataset.flat_map( lambda window: window.batch(window_length) )
dataset = dataset.repeat().batch(1)
dataset = dataset.map( lambda windows: (windows[:, :-1] , windows[:, 1:]) )
dataset = dataset.map(
    lambda X_batch, Y_batch: ( tf.one_hot( X_batch, depth=max_id ), Y_batch ))
dataset = dataset.prefetch(1)

# %% 

model = keras.Sequential([
    keras.layers.GRU( 128, return_sequences=True, stateful=True,
                     batch_input_shape=[1, None, max_id] ),
    keras.layers.GRU( 128, return_sequences=True, stateful=True ),
    keras.layers.TimeDistributed(keras.layers.Dense(max_id, activation='softmax'))
])
'''
model = keras.Sequential([
    keras.layers.GRU( 128, return_sequences=True, input_shape=[None, max_id],
                     dropout=0.2, recurrent_dropout=0.2 ),
    keras.layers.GRU( 128, return_sequences=True,
                     dropout=0.2, recurrent_dropout=0.2 ),
    keras.layers.TimeDistributed(keras.layers.Dense(max_id, activation='softmax'))
])

# no GPU because recurrent_dropouts != 0 ???
'''
# %%

model.compile( loss='sparse_categorical_crossentropy', optimizer='adam' )

# %% 
steps_per_epoch = train_size // 1 // n_steps
history = model.fit( dataset, steps_per_epoch=steps_per_epoch, epochs=10 )

# %% 

# corresponding stateless
stateless_model = keras.Sequential([
    keras.layers.GRU( 128, return_sequences=True, input_shape=[None, max_id] ),
    keras.layers.GRU( 128, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(max_id, activation='softmax'))
])

stateless_model.build( tf.TensorShape( [None, None, max_id] ) )

stateless_model.set_weights( model.get_weights() )
model = stateless_model

# %% 

stateless_model.save( 'models/ch16ShakespreareStatefulRNN_batch1.h5' )
# model = keras.models.load_model('models/ch16ShakespreareStatefulRNN_batch1.h5')

# %% 

t02 = complete_text('t', temperature=0.2)
print('t02: ' + t02)
t1 = complete_text('t', temperature=1.)
print('t1: ' + t1)

# %% batched version

batch_size = 32
encoded_parts = np.array_split( encoded[:train_size], batch_size )
datasets = []
for encoded_part in encoded_parts:
    dataset = tf.data.Dataset.from_tensor_slices( encoded_part )
    dataset = dataset.window(window_length, shift=n_steps, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_length))
    datasets.append( dataset )
dataset = tf.data.Dataset.zip( tuple(datasets) ).map(lambda *windows: tf.stack(windows))
dataset = dataset.repeat().map(lambda windows: (windows[:,:-1], windows[:,1:]))
dataset = dataset.map(
    lambda X_batch, Y_batch: ( tf.one_hot(X_batch, depth=max_id) , Y_batch ) )
dataset.prefetch(1)

# %% 

model = keras.Sequential([
    keras.layers.GRU( 128, return_sequences=True, stateful=True,
                     batch_input_shape=[batch_size, None, max_id] ),
    keras.layers.GRU( 128, return_sequences=True, stateful=True ),
    keras.layers.TimeDistributed(keras.layers.Dense(max_id, activation='softmax'))
])

# %%

model.compile( loss='sparse_categorical_crossentropy', optimizer='adam' )

# %% 
steps_per_epoch = train_size // batch_size // n_steps
history = model.fit( dataset, steps_per_epoch=steps_per_epoch, epochs=10 )

# %% 

# corresponding stateless
stateless_model = keras.Sequential([
    keras.layers.GRU( 128, return_sequences=True, input_shape=[None, max_id] ),
    keras.layers.GRU( 128, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(max_id, activation='softmax'))
])

stateless_model.build( tf.TensorShape( [None, None, max_id] ) )

stateless_model.set_weights( model.get_weights() )
model = stateless_model

# %% 

stateless_model.save( 'models/ch16ShakespreareStatefulRNN_batch32.h5' )
# model = keras.models.load_model('models/ch16ShakespreareStatefulRNN_batch32.h5')

# %% 

t02 = complete_text('t', temperature=0.2)
print('t02: ' + t02)
t1 = complete_text('t', temperature=1.)
print('t1: ' + t1)

# %% sentiment analysis

(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data()
print( repr( X_train[0][:10] ) )

# %% 

word_index = keras.datasets.imdb.get_word_index()

# %%

id_to_word = {id_ + 3: word for word, id_ in word_index.items()}

for id_, token in enumerate( ('<pad>', '<sos>', '<unk>') ):
    id_to_word[id_] = token

s = " ".join( [id_to_word[id_] for id_ in X_train[0][:10]] )
print(s)

# %% preprocessing data example

import tensorflow_datasets as tfds

# %% 

datasets, info = tfds.load('imdb_reviews', as_supervised=True, with_info=True)

# %% 

train_size = info.splits['train'].num_examples

# %% 

def preprocess(X_batch, y_batch):
    # first 300 chars of review are enough
    X_batch = tf.strings.substr(X_batch, 0, 300)
    # replace <br /> with spaces
    X_batch = tf.strings.regex_replace(X_batch, b'<br\\s*/?>', b' ')
    # replace non letters with spaces
    X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z'']", b' ')
    # split by spaces
    X_batch = tf.strings.split( X_batch )
    return X_batch.to_tensor( default_value=b"<pad>" ), y_batch

# %% 

from collections import Counter

vocabulary = Counter()
for X_batch, y_batch in datasets['train'].batch(32).map(preprocess):
    for review in X_batch:
        vocabulary.update( list( review.numpy() ) )

# %% 

print( repr( vocabulary.most_common()[:3] ) )
print('length: ' + repr( len( vocabulary ) ))

# %% 

# keep 10000 words
vocab_size = 10000
truncated_vocabulary = [
    word for word, count in vocabulary.most_common()[:vocab_size]
]

print('length: ' + repr( len( truncated_vocabulary ) ))

# %% 

words = tf.constant(truncated_vocabulary)
word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
num_oov_buckets = 1000
table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)

# %% look up for some phrases

phr1 = b'This movie was faaantastic'
phr2 = b'This movie was faaaantastic'
phr3 = b'This movie was asdfqwerasdf'

print( 'phr1' + repr( table.lookup( tf.constant( [phr1.split()]) ) ) )
print( 'phr2' + repr( table.lookup( tf.constant( [phr2.split()]) ) ) )
print( 'phr3' + repr( table.lookup( tf.constant( [phr3.split()]) ) ) )

# %%

def encode_words( X_batch, y_batch ):
    return table.lookup(X_batch), y_batch

# %% 

train_set = datasets['train'].batch(32).map(preprocess)
train_set = train_set.map(encode_words).prefetch(1)

# %% 

embed_size = 128
model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size, 
                           input_shape=[None]),
    keras.layers.GRU(128, return_sequences=True),
    keras.layers.GRU(128),
    keras.layers.Dense(1, activation='softmax')
])

# %% 

model.compile(loss='binary_crossentropy', optimizer='adam')

# %% 

history = model.fit(train_set, epochs=20)

# %% 

x, _ = encode_words( tf.constant([[b'I', b'liked', b'the', b'movie']]), [0] )
y_pred = model.predict( x )
print( repr( y_pred ) )

# %% 

z = []

for x in datasets['test'].take(5):
    z.append( x )


# %% 

for x in z:
    xx, y = encode_words( tf.constant( [x[0].numpy()] ) , x[1] )
    y_pred = model.predict( xx )
    print( repr( y_pred ) + ' -- ' + repr(y) )


# %% 

import tensorflow_hub as hub

model = keras.Sequential([
    hub.KerasLayer('https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1',
                   dtype=tf.string, input_shape=[], output_shape=50),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# %%

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# %% 

datasets, info = tfds.load('imdb_reviews', as_supervised=True, with_info=True)
train_size = info.splits['train'].num_examples
batch_size = 32
train_set = datasets['train'].batch(batch_size).prefetch(1)
history = model.fit(train_set, epochs=5)

# %% 

z = []

for x in datasets['test'].take(5):
    z.append( x )


# %% 

import tensorflow_addons as tfa

# inputs
encoder_inputs = keras.layers.Input([None], dtype=np.int32)
decoder_inputs = keras.layers.Input([None], dtype=np.int32)
sequence_lengths = keras.layers.Input([], dtype=np.int32)

# embeddings
embeddings = keras.layers.Embedding( vocab_size, embed_size )
encoder_embeddings = embeddings( encoder_inputs )
decoder_embeddings = embeddings( decoder_inputs )

# encoder layer
encoder = keras.layers.LSTM( 512, return_state=True )
encoder_output, state_h, state_c = encoder( encoder_embeddings )
# actual encoder output
encoder_state = [state_h, state_c]

# sampler for defining decoder state during traning and running
sampler = tfa.seq2seq.TrainingSampler()

# decoder
decoder_cell = keras.layers.LSTMCell( 512 )
output_layer = keras.layers.Dense( vocab_size )
decoder = tfa.seq2seq.basic_decoder.BasicDecoder(decoder_cell, sampler,
                                                 output_layer=output_layer)

# output of decoder
final_outputs, final_state, final_sequence_lengths = decoder(
    decoder_embeddings, initial_state=encoder_state,
    sequence_length=sequence_lengths
)

# output of the network
Y_proba = tf.nn.softmax( final_outputs.rnn_output )

# model
model = keras.Model( inputs=[encoder_inputs, decoder_inputs, sequence_lengths],
                    outputs = [Y_proba] )

# %%

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# %%

X = np.random.randint(100, size=10*1000).reshape(1000,10)
Y = np.random.randint(100, size=15*1000).reshape(1000,15)

X_decoder = np.c_[np.zeros((1000, 1)), Y[:, :-1]]

seq_lengths = np.full([1000], 15)

# %% 

history = model.fit([X, X_decoder, seq_lengths], Y, epochs=2)

# %% 

class PositionalEncoding(keras.layers.Layer):
    def __init__( self, max_steps, max_dims, dtype=tf.float32, **kwargs ):
        super().__init__( dtype=dtype, **kwargs )
        if max_dims%2 == 1: max_dims += 1
        p,i = np.meshgrid(np.arange(max_steps), np.arange(max_dims//2))
        pos_emb = np.empty((1, max_steps, max_dims))
        pos_emb[0, :, ::2] = np.sin(p/10000**(2*i/max_dims)).T
        pos_emb[0, :, 1::2] = np.cos(p/10000**(2*i/max_dims)).T
        self.positional_encoding = tf.constant(pos_emb.astype(self.dtype))
    def call(self, inputs):
        shape = tf.shape(inputs)
        return inputs + self.positional_encoding[:, :shape[-2], :shape[-1]]

# %% 

max_steps = 201
max_dims = 512
pos_emb = PositionalEncoding(max_steps, max_dims)
PE = pos_emb( np.zeros((1 ,max_steps, max_dims) , np.float32) )[0].numpy()

# %% 

embed_size = 512; max_steps = 500; vocab_size = 10000

encoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
decoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)

embeddings = keras.layers.Embedding(vocab_size, embed_size)
encoderd_embeddings = embeddings(encoder_inputs)
decoderd_embeddings = embeddings(decoder_inputs)

positional_encoding = PositionalEncoding(max_steps, max_dims=embed_size)
encoder_in = positional_encoding(encoder_embeddings)
decoder_in = positional_encoding(decoder_embeddings)

# %%



''' 

# %% lambda example

l = [1,2,3,4]

y = map( lambda x: x**2 + 2, l )

# %% dataset batch example

d1 = tf.data.Dataset.from_tensor_slices( [1,2,3,4,5,6,7,8] )
print( 'before batching: ' + repr( list( d1.as_numpy_iterator() ) ) )
d2 = d1.batch(3)
print( 'after batching: ' + repr( list( d2.as_numpy_iterator() ) ) )
d3 = d2.shuffle(2)
print( 'batched shuffled: ' + repr( list( d3.as_numpy_iterator() ) ) )
d4 = d1.shuffle(2)
print( 'unbatched shuffled: ' + repr( list( d4.as_numpy_iterator() ) ) )
d5 = d1.window( 3, shift=1, drop_remainder=True )
for window in d5:
    print( 'unbatched windowed: ' + repr( list( window.as_numpy_iterator() ) ) )
d6 = d5.flat_map( lambda w: w.batch( 3 ) )
print( 'flat: ' + repr( list( d6.as_numpy_iterator() ) ) )

# %% 

# without reshuffle_each_iteration=False ...

d7 = d6.shuffle(100, reshuffle_each_iteration=False).batch( 3 )
# d7 = d6.shuffle(100)
print( 'shuffled - batched: ' + repr( list( d7.as_numpy_iterator() ) ) )
d8 = d7.map( lambda w: (w[:,:-1], w[:,1:]) )
print( 'mapped: ' + repr( list( d8.as_numpy_iterator() ) ) )
# ... even within one print, calls to d8 get changed so below we would see
# inconsistent results
print( repr( list(d8)[0][0] ) + ' \n ' + repr( list(d8)[0][1] ) )
print( repr( list(d8)[1][0] ) + ' \n ' + repr( list(d8)[1][1] ) )


# %%

d9 = d8.map(
    lambda X_batch, Y_batch: ( tf.one_hot(X_batch, depth=8), Y_batch ) )

print( 'one hot: ' + repr( list( d9.as_numpy_iterator() ) ) )



'''