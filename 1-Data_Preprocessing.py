#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:18:20 2018

@author: ahmet
"""

#importing libraries and dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv("dataset.csv")

height = data[["height"]]
height_weight = data[["height","weight"]]

#%%missing values
from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values="NaN", strategy = "mean", axis=0 )    

numbers = data.iloc[:,1:4].values
numbers = imputer.fit_transform(numbers)
#%%encoder:  categorical to numeric

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
ohe = OneHotEncoder(categorical_features="all")

country = data.iloc[:,0:1].values
country[:,0] = le.fit_transform(country[:,0])
country = ohe.fit_transform(country).toarray()

#%%dataframe
country_df = pd.DataFrame(data = country, index = range(22), 
                     columns=["fr","tr","us"] )


numbers_df = pd.DataFrame(data = numbers, index = range(22),
                     columns = ["height","weight","age"])


gender = data.iloc[:,-1].values


gender_df = pd.DataFrame(data = gender , index=range(22), columns=["gender"])

#%%concating
data_new = pd.concat([country_df,numbers_df],axis=1)
data_all = pd.concat([data_new,gender_df],axis=1)

#%%data split into train and test
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(
        data_new,gender_df,test_size=0.33, random_state=0)

#%%feature scaling
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
X_train = scale.fit_transform(x_train)
X_test = scale.fit_transform(x_test)






    
    

