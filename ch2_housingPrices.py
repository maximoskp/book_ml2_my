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

def test_set_check( identifier , test_ratio ):
    return crc32( np.int64( identifier ) ) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id( data , test_ratio , id_column ):
    ids = data[id_column]
    in_test_set = ids.apply( lambda id_ : test_set_check( id_ , test_ratio ) )
    return data.loc[ ~in_test_set ] , data.loc[ in_test_set ]

# id based on item's row number
housing_with_id = housing.reset_index() # add index column
train_set , test_set = split_train_test_by_id( housing_with_id , 0.2 , "index" )

# but if data are mixed with others, then row numbers get mixed
# so it's a good idea to link id to a constant property, e.g. geographic location
# e.g. group by id based on a (constant) combination of longitude and latitude
housing_with_id[ "id" ] = housing[ "longitude" ] * 1000 + housing[ "latitude" ]
train_set , test_set = split_train_test_by_id( housing_with_id , 0.2 , "id" )

# or use sklearn's train-test split that is based on given random seed
from sklearn.model_selection import train_test_split
train_test , test_set = train_test_split( housing , test_size=0.2 , random_state=42 )

# take stratified sample based on income levels
# so create income levels
housing[ "income_cat" ] = pd.cut( housing["median_income"] ,
                                 bins = [0., 1.5, 3., 4.5, 6., np.inf],
                                 labels = [1,2,3,4,5] )
# and show histogram of strata
housing['income_cat'].hist()

# get stratified sample
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit( n_splits = 1, test_size = 0.2, random_state = 42)

for train_index , test_index in split.split( housing , housing['income_cat'] ):
    strat_train_set = housing.loc[ train_index ]
    strat_test_set = housing.loc[ test_index ]

# check stratification results
real_overall_strat = housing['income_cat'].value_counts() / len(housing)
real_train_strat = strat_train_set['income_cat'].value_counts() / len(strat_train_set)
real_test_strat = strat_test_set['income_cat'].value_counts() / len(strat_test_set)

# ploting geography of data
housing.plot(kind = 'scatter', x='longitude' , y='latitude', alpha=0.1)
# marker size (s) represents district population
# color (c) represents price
housing.plot( kind = 'scatter', x='longitude' , y='latitude', alpha=0.4,
             s = housing['population']/100, label='population', figsize=(10,7),
             c = 'median_house_value', cmap=plt.get_cmap('jet'), colorbar=True )
plt.legend()
plt.savefig('figs/ch2_geography.png', dps=300)

# check for correlations inside the data - all pairs of columns
corr_matrix = housing.corr()
corr_matrix[ corr_matrix >= 0.5 ]