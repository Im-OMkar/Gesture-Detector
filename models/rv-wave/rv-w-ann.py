# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 14:51:05 2020

@author: Omkar
"""
# Artificial Neural Network

"""import numpy as np
import matplotlib.pyplot as plt"""
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('rv-wave.csv')
X = dataset.iloc[:, [0,1,2]].values
y = dataset.iloc[:, 3].values

"""# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]"""

# Splitting into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

########################
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# input layer and the first hidden layer
classifier.add(Dense( units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 3))

#second hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu'))

#  the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# Predicting
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Accuracy 
from sklearn.metrics import accuracy_score
score =accuracy_score(y_test,y_pred)