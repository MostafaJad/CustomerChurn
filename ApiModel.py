#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 20:14:13 2019

@author: jado
"""

import numpy as np
import pandas as pd
  

pd.set_option('display.max_columns',30) # set the maximum width
# Load the dataset in a dataframe object 
df = pd.read_csv('/Users/jado/Desktop/DataWarhousingProject/Telco-Customer-Churn.csv')

print(df.columns.values)
print (df.head())
categories = []
for col, col_type in df.dtypes.iteritems():
     if col_type == 'O':
          categories.append(col)
     else:
          df[col].fillna(0, inplace=True)
print(categories)
print(df.columns.values)
print(df.head())
df.describe()
df.dtypes
#check for null values
print(len(df) - df.count())  #Cabin , boat, home.dest have so many missing values



include = ['MonthlyCharges','gender', 'Dependents', 'Churn']
df_ = df[include]
print(df_.columns.values)
print(df_.head())
df_.describe()
df_.dtypes
df_['MonthlyCharges'].unique()
df_['gender'].unique()
df_['Dependents'].unique()
df_['Churn'].unique()
# check the null values
print(df_.isnull().sum())
print(df_['gender'].isnull().sum())
print(df_['Dependents'].isnull().sum())
print(len(df_) - df_.count())





df_.dropna(axis=0,how='any',inplace=True)  



categoricals = []
for col, col_type in df_.dtypes.iteritems():
     if col_type == 'O':
          categoricals.append(col)
     else:
          df_[col].fillna(0, inplace=True)
print(categoricals)





df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=False)
pd.set_option('display.max_columns',30)
print(df_ohe.head())
print(df_ohe.columns.values)
print(len(df_ohe) - df_ohe.count())





from sklearn import preprocessing
# Get column names first
names = df_ohe.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df_ohe)
scaled_df = pd.DataFrame(scaled_df, columns=names)
print(scaled_df.head())
print(scaled_df['MonthlyCharges'].describe())
print(scaled_df['gender_Male'].describe())
print(scaled_df['gender_Female'].describe())
print(scaled_df['Dependents_Yes'].describe())
print(scaled_df['Dependents_No'].describe())
print(scaled_df['Churn_No'].describe())
print(scaled_df['Churn_Yes'].describe())

print(scaled_df.dtypes)





from sklearn.linear_model import LogisticRegression
dependent_variable = 'MonthlyCharges'
# Another way to split the three features
x = scaled_df[scaled_df.columns.difference([dependent_variable])]
x.dtypes
y = scaled_df[dependent_variable]
#convert the class back into integer
y = y.astype(int)
# Split the data into train test
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(x,y, test_size = 0.2)
#build the model
lr = LogisticRegression(solver='lbfgs')
lr.fit(x, y)
# Score the model using 10 fold cross validation
from sklearn.model_selection import KFold
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(lr, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
print ('The score of the 10 fold run is: ',score)





testY_predict = lr.predict(testX)
testY_predict.dtype
#print(testY_predict)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics 
labels = y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels))






import joblib 
joblib.dump(lr, '/Users/jado/Desktop/DataWarhousingProject/model_lr2.pkl')
print("Model dumped!")





model_columns = list(x.columns)
print(model_columns)
joblib.dump(model_columns, '/Users/jado/Desktop/DataWarhousingProject/model_columns.pkl')
print("Models columns dumped!")




  







