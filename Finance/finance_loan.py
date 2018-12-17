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

dataset2.corrwith(dataset.e_signed).plot.bar(figsize = (20,10), title = "Correlation with E-signed", fontsize = 6, rot = 45, grid = True)

###### Correlation Matrix
sn.set(style="white")

# Computing the correlation matrix
corr = dataset2.corr()

# Generating a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Setting up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))

# Generating a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Drawing the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


import random
import time

random.seed(100)

### Feature Engineering
dataset = dataset.drop(columns = ["months_employed"]) 
dataset["personal_account_months"] = (dataset.personal_account_m + (dataset.personal_account_y * 12))
dataset[['personal_account_m', 'personal_account_y', 'personal_account_months']].head() 
dataset = dataset.drop(columns = ["personal_account_m", "personal_account_y"])

##### One Hot Encoding
dataset = pd.get_dummies(dataset)
dataset.columns
dataset = dataset.drop(columns = ["pay_schedule_semi-monthly"])

response = dataset['e_signed']
users = dataset['entry_id']
dataset = dataset.drop(columns = ['e_signed', 'entry_id'])

# Splitting into Trainng set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, response, test_size=0.2, random_state=0)

####   Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2

##### Comparing models 

# Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0, penalty='l1')
classifier.fit(X_train, y_train)
# predicting the test set
y_pred = classifier.predict(X_test)
# determining the accurancy and performance
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
acc = accuracy_score(y_test, y_pred)
prec =  precision_score(y_test, y_pred)
rec =  recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
# store the results in a pandas dataframe
results = pd.DataFrame([['Linear Regression (lasso)', acc, prec, rec, f1]], 
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

## SVM (Linear)
from sklearn.svm import SVC
classifier = SVC(random_state=0, kernel = 'linear') 
classifier.fit(X_train, y_train)
# predicting the test set
y_pred = classifier.predict(X_test)
# determining the accurancy and performance
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
acc = accuracy_score(y_test, y_pred)
prec =  precision_score(y_test, y_pred)
rec =  recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
# store the results in a pandas dataframe
model_results = pd.DataFrame([['SVM (Linear)', acc, prec, rec, f1]], 
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
# append the model results of the SVM to to the result dataframe
results = results.append(model_results, ignore_index = True)

## SVM (Gaussian)
from sklearn.svm import SVC
classifier = SVC(random_state=0, kernel = 'rbf')
classifier.fit(X_train, y_train)
# predicting the test set
y_pred = classifier.predict(X_test)
# determining the accurancy and performance
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
acc = accuracy_score(y_test, y_pred)
prec =  precision_score(y_test, y_pred)
rec =  recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
# store the results in a pandas dataframe
model_results = pd.DataFrame([['SVM (Gaussian rbf)', acc, prec, rec, f1]], 
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
# append the model results of the SVM to to the result dataframe
results = results.append(model_results, ignore_index = True)
