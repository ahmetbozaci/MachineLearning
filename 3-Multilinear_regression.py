# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 14:29:25 2018

@author: Ahmet
"""

#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("dataset.csv")

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)

numbers = data.iloc[:,1:4].values
imputer = imputer.fit(numbers)
numbers = imputer.transform(numbers)

#categorical 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

le = LabelEncoder()
ohe = OneHotEncoder(categorical_features="all") 

country = data.iloc[:,0:1].values
country[:,0] = le.fit_transform(country[:,0])
country = ohe.fit_transform(country).toarray()

gender = data.iloc[:,-1:].values
gender[:,0] = le.fit_transform(gender[:,0])
gender = ohe.fit_transform(gender).toarray()

#dataframe 
country_df = pd.DataFrame(data = country, index = range(22), 
                     columns = ["fr", "tr", "us"])
numbers_df = pd.DataFrame(data = numbers, index = range(22), 
                      columns = ["height","weight","age"])    


gender_df = pd.DataFrame(data = gender[:,:1], index = range(22), 
                      columns = ["gender"])

#concatenating
data1 = pd.concat([country_df,numbers_df],axis=1)
print(data)
data_all = pd.concat([data1,gender_df],axis=1)
print(data_all)

#testing
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        data1, gender_df, test_size = 0.33, random_state = 0 )

#Linear regression
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train)

y_pred = linear_reg.predict(x_train)

height = data_all.iloc[:,3:4].values

data_split = pd.concat([data_all.iloc[:,:3], data_all.iloc[:,4:]], axis=1)

x_train, x_test, y_train, y_test = train_test_split(
        data_split,height, test_size = 0.33, random_state = 0 )

linear_reg2 = LinearRegression()
linear_reg2.fit(x_train, y_train)

#backward elimination
import statsmodels.api as sm

#x = np.append(arr = np.ones((22,1)).astype(int), values=veri, axis=1 )
x_list = data_split.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog = height, exog = x_list).fit()
print(r_ols.summary())

x_list = data_split.iloc[:,[0,1,2,3,5]].values
r_ols = sm.OLS(endog = height, exog = x_list).fit()
print(r_ols.summary())

x_list = data_split.iloc[:,[0,1,2,3]].values
r_ols = sm.OLS(endog = height, exog = x_list).fit()
print(r_ols.summary())

#r2 score
from sklearn.metrics import r2_score
print("Multiregresyon R2:", r2_score(y_train,linear_reg2.predict(x_train)))
