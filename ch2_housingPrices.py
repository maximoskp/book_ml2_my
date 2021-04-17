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

# %%

# show info of all data frame
description = housing.describe()

# plot histograms for dataframe
plt.clf()
housing.hist(bins=50, figsize=(20,15))
plt.savefig( os.path.join('figs', 'ch2_hists.png') , dpi=300 )

# %%

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

# %%

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

# %%

# but if data are mixed with others, then row numbers get mixed
# so it's a good idea to link id to a constant property, e.g. geographic location
# e.g. group by id based on a (constant) combination of longitude and latitude
housing_with_id[ "id" ] = housing[ "longitude" ] * 1000 + housing[ "latitude" ]
train_set , test_set = split_train_test_by_id( housing_with_id , 0.2 , "id" )

# %%

# or use sklearn's train-test split that is based on given random seed
from sklearn.model_selection import train_test_split
train_set_check , test_set = train_test_split( housing , test_size=0.2 , random_state=42 )

# %%

# take stratified sample based on income levels
# so create income levels
housing[ "income_cat" ] = pd.cut( housing["median_income"] ,
                                 bins = [0., 1.5, 3., 4.5, 6., np.inf],
                                 labels = [1,2,3,4,5] )
# and show histogram of strata
housing['income_cat'].hist()

# %%

# get stratified sample
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit( n_splits = 1, test_size = 0.2, random_state = 42)

# %%

for train_index , test_index in split.split( housing , housing['income_cat'] ):
    strat_train_set = housing.loc[ train_index ]
    strat_test_set = housing.loc[ test_index ]

# %%

# check stratification results
real_overall_strat = housing['income_cat'].value_counts() / len(housing)
real_train_strat = strat_train_set['income_cat'].value_counts() / len(strat_train_set)
real_test_strat = strat_test_set['income_cat'].value_counts() / len(strat_test_set)

# %%

# ploting geography of data
housing.plot(kind = 'scatter', x='longitude' , y='latitude', alpha=0.1)
# marker size (s) represents district population
# color (c) represents price
housing.plot( kind = 'scatter', x='longitude' , y='latitude', alpha=0.4,
             s = housing['population']/100, label='population', figsize=(10,7),
             c = 'median_house_value', cmap=plt.get_cmap('jet'), colorbar=True )
plt.legend()
plt.savefig('figs/ch2_geography.png', dpi=300)

# %%

# check for correlations inside the data - all pairs of columns
corr_matrix = housing.corr()
# all correlations for a specific attribute
median_value_ascending_corrs = corr_matrix['median_house_value'].sort_values(ascending=False)

# %%

# plot pairs of values for visualisation of possible correlations
from pandas.plotting import scatter_matrix

attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix( housing[attributes] , figsize=(12,8) )

# %%

# check compound correlations by creating new compound attributes
housing['bedrooms_per_room']  = housing['total_bedrooms']/housing['total_rooms']
corr_matrix = housing.corr()
# all correlations for a specific attribute
median_value_ascending_corrs = corr_matrix['median_house_value'].sort_values(ascending=False)

# %% 

# prepare data for machine learning
# separate predictors and labels
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()

# %%

# there are missing values in, e.g., total_bedrooms we can either:
# 1) drop the entire attribute/column
# housing.drop('total_bedrooms', axis=1)
# 
# 2) get rid of the entire districts/rows
# housing.drop(subset=['total_bedrooms'])
# 
# 3) set missing values to specific value, e.g. median
median = housing['total_bedrooms'].median()
housing['total_bedrooms'].fillna(median, inplace=True)

# %%

# or we can use scikit-learns' SimpleImputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')

# %%

# since this only works in numerical attributes, we need to drop nominal attributes
housing_num = housing.drop('ocean_proximity', axis=1)
imputer.fit(housing_num)

# check statistics and validate that they match with data
impustats = imputer.statistics_
houstats = housing_num.median().values
print('imputer.statistics_: ' + repr(impustats))
print('housing_num.median(): ' + repr(houstats))

# %%

# use imputer to transform data
X = imputer.transform( housing_num )
# and create new dataframe
housing_tr = pd.DataFrame( X , columns=housing_num.columns , index=housing_num.index )

# %%

# handling text and categorical attributes
# e.g. ocean proximity
housing_cat = housing[['ocean_proximity']]
# check
print(repr(housing_cat.head(10)))

# %% 
# numerical transformation through 'random' ordinal number assignment
# not always useful, since selected ordinal encoding might not be suitable
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(repr(housing_cat_encoded[:10]))
# check categories
print(ordinal_encoder.categories_)

# %%

# or, more properly, one hot encoding
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(repr(housing_cat_1hot[:10].toarray()))
# check categories
print(cat_encoder.categories_)

# %%

# we can also build a custom transformer (with duck typing) for adding
# compound features

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_houshold = X[:, rooms_ix]/X[:,households_ix]
        population_per_houshold = X[:, population_ix]/X[:,households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X, rooms_per_houshold, population_per_houshold, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_houshold, population_per_houshold]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=True)
housing_extra_attribs = attr_adder.transform(housing.values)

# comment: np.c_ easy column concatenation - there is also np.r_

# %% 

# feature scaling / normalisation
# normalisation: put in range [0,1] with MinMaxScaler
# zero mean - one std with StandardScaler (less affected by outliers)

# transformation pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler())
    ])

housing_num_tr = num_pipeline.fit_transform( housing_num )

# %% 

# for handling categorical and numerical data

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', OneHotEncoder(), cat_attribs)
    ])

housing_prepared = full_pipeline.fit_transform(housing)

# %%

# select training model

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit( housing_prepared , housing_labels )

# %%

# test model on training data
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform( some_data )

# %%

linear_predictions = lin_reg.predict( some_data_prepared )
print('predictions: \n' + repr(list(linear_predictions)) )
print('labels: \n' + repr(list(some_labels)) )

# %% check accuracy with metrics

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict( housing_prepared )
lin_mse = mean_squared_error( housing_labels, housing_predictions )
lin_rmse = np.sqrt( lin_mse )
print( 'linear regression rmse: ' + str( lin_rmse ) )

# %% try decision tree

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit( housing_prepared, housing_labels )

housing_predictions = tree_reg.predict( housing_prepared )
tree_mse = mean_squared_error( housing_labels, housing_predictions )
tree_rmse = np.sqrt( tree_mse )
print( 'tree regression rmse: ' + str( tree_rmse ) )

# %% applying 10-fold cross validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score( lin_reg, housing_prepared, housing_labels,
                         scoring='neg_mean_squared_error', cv=10 )
lin_scores = np.sqrt( -scores )

scores = cross_val_score( tree_reg, housing_prepared, housing_labels,
                         scoring='neg_mean_squared_error', cv=10 )
tree_scores = np.sqrt( -scores )

def display_scores( scores ):
    # print( 'scores: ' + repr(scores) )
    print( 'mean: ' + str(scores.mean()) )
    print( 'std: ' + str(scores.std()) )

print('linear regression:')
display_scores(lin_scores)
print('tree regression:')
display_scores(tree_scores)

# %% random forest

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit( housing_prepared , housing_labels )

scores = cross_val_score( forest_reg , housing_prepared , housing_labels,
                                     scoring='neg_mean_squared_error', cv=10)
forest_scores = np.sqrt( -scores )
print('forest regression')
display_scores( forest_scores )

# %% easy way to save model

import joblib

joblib.dump( forest_reg , 'saved_models/ch2_rand_forest_reg_model.pkl' )
# load as: forest_reg = joblib.load('saved_models/ch2_rand_forest_reg_model.pkl')