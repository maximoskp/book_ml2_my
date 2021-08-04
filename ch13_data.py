#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 15:32:56 2021

@author: max
"""

import tensorflow as tf

# %% create dataset from tensor

X = tf.range(10)
dataset = tf.data.Dataset.from_tensor_slices(X)
print(dataset)

# iterate
for item in dataset:
    print( item )

# %% chaining

dataset = dataset.repeat(3).batch(7)
for item in dataset:
    print( item )

# repeat does not store data in RAM, it's a simple instruction. Calling it
# with no arguments will be considered as repeating the dataset forever
# 'drop_remainder'=True in batch command to leave final batch out


# %% apply lambda to transform
dataset = dataset.map(lambda x: x**2)
for item in dataset:
    print( item )

# %% unbatch and filter

dataset = dataset.unbatch()
dataset = dataset.filter( lambda x: x < 7 )
for item in dataset:
    print( item )

# %% take portion

dataset = dataset.take(3)
for item in dataset:
    print( item )

# %% create again and shuffle

dataset = tf.data.Dataset.range(10).repeat(3).shuffle(buffer_size=3).batch(7)
for item in dataset:
    print( item )

# %% embeddings

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_mean = scaler.mean_
X_std = scaler.scale_

# %%

vocab = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
indices = tf.range( len(vocab) , dtype=tf.int64 )

table_init = tf.lookup.KeyValueTensorInitializer(vocab, indices)
num_oov_buckets = 2
table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)

# %% example run

categories = tf.constant(['NEAR BAY', 'DESERT', 'INLAND', 'INLAND'])
cat_indices = table.lookup( categories )
print('cat_indices: ' + repr(cat_indices))

cat_one_hot = tf.one_hot(cat_indices, depth=len(vocab)+num_oov_buckets)
print('cat_one_hot: ' + repr(cat_one_hot))

# %% 

embedding_dim = 2
embed_init = tf.random.uniform( [len(vocab) + num_oov_buckets, embedding_dim] )
embedding_matrix = tf.Variable( embed_init )

# %% from categories to embeddings

cat_embed = tf.nn.embedding_lookup(embedding_matrix, cat_indices)
print('cat_embed: ' + repr(cat_embed))

# %% create layer

from tensorflow import keras

embedding = keras.layers.Embedding( input_dim=len(vocab)+num_oov_buckets,
                                   output_dim=embedding_dim)

print(embedding( cat_indices ))
