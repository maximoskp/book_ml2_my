#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 17:37:41 2020

@author: max
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

HOUSING_PATH = os.path.join('datasets', 'housing')

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)

housing = load_housing_data()

# show info of all data frame
description = housing.describe()

# plot histograms for dataframe
plt.clf()
housing.hist(bins=50, figsize=(20,15))
plt.savefig( os.path.join('figs', 'ch2_hists.png') , dpi=300 )

# check ocean_proximity, which is a non-numeric value
ocean_proximity = housing['ocean_proximity'].value_counts()

# split train and test sets
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices] , data.iloc[test_indices]

train_set , test_set = split_train_test(housing, 0.2)

# keep this split for all other runs (?)
from zlib import crc32