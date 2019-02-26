#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 22:00:12 2019

@author: oscar
"""

## Part 1 data preprocessign


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()  # for geograpghy
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])  # found in index 1 of the X dataset
labelencoder_X_2 = LabelEncoder()  # for gender
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])  # found in index 2 of the X dataset
onehotencoder = OneHotEncoder(categorical_features = [1])  # for dummy variables in index 1
X = onehotencoder.fit_transform(X).toarray()
# to prevent the dummy trap we delete one index from the geography column 
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

### part 2 BUilding the ANN

# import keras packages
import keras
from keras.models import Sequential # used to initialise the NN
from keras.layers import Dense #  used to create the layers in the NN
from keras.layers import Dropout # used to reduce overfitting 

#### NOTE:  Dropout Regularisation is used in deep learning to reduce the challenge of over fitting
# if you have a high variance of your model, then you have over fitting

########################################################################################################

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
# add dropout to reduce over fitting in this layer
classifier.add(Dropout(p = 0.1)) # never go beyond p = 0.5 (Under fitting)

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

########################################################################################################

#Evaluating the ANN
import keras
from keras.models import Sequential # used to initialise the NN
from keras.layers import Dense #  used to create the layers in the NN
from keras.layers import Dropout # used to reduce overfitting 
from keras.wrappers.scikit_learn import KerasClassifier # used to run keras with scikit learn 
from sklearn.model_selection import cross_val_score # used to caluculate K-fold validation
from sklearn.model_selection import GridSearchCV # used for parameter tuning with Grid Search algorithm

#### NOTE:  Dropout Regularisation is used in deep learning to reduce the challenge of over fitting
# if you have a high variance of your model, then you have over fitting

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
# create a new classifier trained on K-fold validation
classifier = KerasClassifier(build_fn = 'build_classifier', batch_size = 10, epochs = 100)
# create a K-fold validation for 10 accuracies
accuracies = cross_val_score(estimator = 'classifier', X = X_train, y = y_train, cv=10, n_jobs = -1)
# mean 
mean = accuracies.mean()
#variance 
variance = accuracies.std()

# Improving the ANN
#### NOTE:  Dropout Regularisation is used in deep learning to reduce the challenge of over fitting
# if you have a high variance of your model, then you have over fitting

########################################################################################################

## Parameter tuning with Grid Search
import keras
from keras.models import Sequential # used to initialise the NN
from keras.layers import Dense #  used to create the layers in the NN
from keras.layers import Dropout # used to reduce overfitting 
from keras.wrappers.scikit_learn import KerasClassifier # used to run keras with scikit learn 
from sklearn.model_selection import GridSearchCV # used for parameter tuning with Grid Search algorithm

#### NOTE:  Dropout Regularisation is used in deep learning to reduce the challenge of over fitting
# if you have a high variance of your model, then you have over fitting

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# create a new classifier for parameter tuning
classifier = KerasClassifier(build_fn = 'build_classifier')
# create a dictionary to hold our values that we want to tune
parameters = {'batch_size' : [25, 32], 
              'epochs' : [100, 500],
              'optimizer' : ['adam', 'rmsprop']}
# create the grid search objective 
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_param
best_accuracy = grid_search.best_score_