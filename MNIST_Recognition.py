#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 21:32:03 2020

@author: anup
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%pylab inline


#%% Read Train Data

train_df = pd.read_csv('train.csv')


#%% Perform EDA

print("Number of rows %s" % train_df.shape[0]) #Print number of rows

print("Number of columns %s" % train_df.shape[1]) #Print number of columns

train_df.head() #Print first 5 rows

train_df.columns #Print Column names

train_df.isnull().values.any() #Print any missing values

train_df.label.unique() #Since label is the target variable, print the unique values in the dataset



#%% Data PreProcessing and Visualization
# split dataset train and validation

from sklearn.model_selection import train_test_split

train_df_X =  train_df.drop(columns=['label'],axis=1)
train_df_Y = train_df['label']


plt.figure(figsize=(15,15))
for num in range(0,40):
    plt.subplot(8,8,num+1)
    grid_data = train_df_X.iloc[num].to_numpy().reshape(28,28)  # reshape from 1d to 2d pixel array
    plt.gca().set_title(train_df_Y[num])
    plt.imshow(grid_data, interpolation = "none", cmap = "bone_r")
    xticks([])
    yticks([])



train_X, val_X, train_Y, val_Y = train_test_split(train_df_X,train_df_Y,test_size=0.2,random_state=123)

# Reduce the data using PCA

from sklearn.decomposition import PCA

pca = PCA(2)
#Components 2 were selected just for the better visualization. 
pca.fit(train_X)
train_X_transformed = pca.transform(train_X)

figure(figsize(6,5))
plt.scatter(train_X_transformed[:,0],train_X_transformed[:,1], s=20, c = train_Y, cmap = "nipy_spectral", edgecolor = "None")
plt.colorbar()
clim(0,9)

xlabel("PC1")
ylabel("PC2")

#Upon plotting we can clearly see the clusters with the dataset. But transforming into higher components may give better clustering.

pca = PCA(100)
pca.fit(train_X)

train_X_transformed = pca.transform(train_X)

val_X_transformed = pca.transform(val_X)

#%% Develop KNN model for classification

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(train_X_transformed,train_Y)

#%% Validate the KNN prediction

from sklearn import metrics

val_Y_pred = knn.predict(val_X_transformed)

acc = metrics.accuracy_score(val_Y,val_Y_pred)

print("KNN accuracy is :" , acc) # accuracy is around 97.0%


#%% Save the trained model

import pickle

knnPkl = open('MNIST_KNN_model_97.0' , 'wb')

pickle.dump(knn, knnPkl)


#%% Let's Predict the test data set

test_df = pd.read_csv('test.csv')

plt.figure(figsize=(5,5))
for digit_num in range(0,40):
    plt.subplot(8,8,digit_num+1)
    grid_data = test_df.iloc[digit_num].to_numpy().reshape(28,28)  # reshape from 1d to 2d pixel array
    plt.imshow(grid_data, interpolation = "none", cmap = "bone_r")
    xticks([])
    yticks([])


test_X_transformed = pca.transform(test_df)

test_Y_Pred = knn.predict(test_X_transformed)

#Visualize the first 40 images
plt.figure(figsize=(15,15))
for num in range(0,40):
    plt.subplot(8,8,num+1)
    grid_data = test_df.iloc[num].to_numpy().reshape(28,28)  # reshape from 1d to 2d pixel array
    plt.gca().set_title(test_Y_Pred[num])
    plt.imshow(grid_data,interpolation = "none", cmap = "bone_r")
    xticks([])
    yticks([])

#Save the results
np.savetxt('knnresults.csv', 
           np.c_[range(1,len(test)+1),test_Y_Pred], 
           delimiter=',', 
           header = 'Id,Label', 
           comments = '', 
           fmt='%d')


#%% Train SVM Classifier

from sklearn import svm

svm = svm.SVC()

svm.fit(train_X_transformed,train_Y)

val_Y_pred = svm.predict(val_X_transformed)

acc = metrics.accuracy_score(val_Y,val_Y_pred)

print("SVM accuracy is :" , acc) # accuracy is around 98.1%


#%% Save the trained model

import pickle

svmPkl = open('MNIST_SVM_model_98.1' , 'wb')

pickle.dump(svm, svmPkl)


#%% Predict of Test Data


test_Y_Pred = svm.predict(test_X_transformed)


np.savetxt('svmresults.csv', 
           np.c_[range(1,len(test)+1),test_Y_Pred], 
           delimiter=',', 
           header = 'Id,Label', 
           comments = '', 
           fmt='%d')

#Visualize the first 40 images
plt.figure(figsize=(15,15))
for num in range(0,40):
    plt.subplot(8,8,num+1)
    grid_data = test_df.iloc[num].to_numpy().reshape(28,28)  # reshape from 1d to 2d pixel array
    plt.gca().set_title(test_Y_Pred[num])
    plt.imshow(grid_data,interpolation = "none", cmap = "bone_r")
    xticks([])
    yticks([])

