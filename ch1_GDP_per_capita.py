#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 15:01:09 2020

@author: max
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import os

def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

# load data
oecd_bli = pd.read_csv('datasets' + os.sep + 'lifesat' + os.sep + 'oecd_bli_2015.csv', thousands=',')
gdp_per_capita = pd.read_csv('datasets' + os.sep + 'lifesat' + os.sep + 'gdp_per_capita.csv', thousands=',', delimiter='\t', encoding='latin1')

country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)

x = np.c_[country_stats['GDP per capita']]
y = np.c_[country_stats['Life satisfaction']]

# linear regression model
model = sklearn.linear_model.LinearRegression()
model.fit(x,y)
# fit Cyprus
x_new = [[22587]] # GDP per capita
# make prediction
y_new = model.predict(x_new)

# visualise the data
plt.clf()
country_stats.plot( kind='scatter', x='GDP per capita', y='Life satisfaction' )
plt.plot(x_new[0][0], y_new[0][0], 'rx')
plt.savefig('figs' + os.sep + 'ch1_gdp_satisfaction.png', dpi=300)