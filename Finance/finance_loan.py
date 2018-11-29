#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 19:40:33 2018

@author: oscar
"""

#importing the libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

# import the dataset
dataset = pd.read_csv('financial_data.csv')

# visualising the content of the dataset
dataset.head()

# visualising the statistics of the content in the dataset
dataset.describe()

# Cleaning the data
# removing NaN
dataset.isna().any() # this code checks if there is any missing data in the dataset # no NaNs

# Plotting the Histogram

# create a new dataset without e_signed(prediction variable), entry_id since it carries no value and pay_schedule column(categorical variable)
dataset2 = dataset.drop(columns = ['e_signed', 'entry_id', 'pay_schedule'])

# Plotting individual numerical columns in the dataset
fig = plt.figure(figsize=(15, 12))  #setting the figure size
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
# for each of the columns, create a subplot for it
for i in range(dataset2.shape[1]):
    plt.subplot(6, 3, i + 1)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i])

    vals = np.size(dataset2.iloc[:, i].unique())
    if vals >= 100:
        vals = 100
    
    plt.hist(dataset2.iloc[:, i], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # laying out the rectangle
