# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:56:50 2018

@author: Ahmet
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("tennis.csv")

#encoder
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

data = dataset.apply(LabelEncoder().fit_transform)

ohe = OneHotEncoder()
weather = data.iloc[:,:1]
weather = ohe.fit_transform(weather).toarray()

#concat
outlook = pd.DataFrame(data = weather, index = range(14), 
                          columns = ["overcast","rainy","sunny"])
data_new = pd.concat([outlook, dataset.iloc[:,1:3]], axis=1)
data_new = pd.concat([data_new, data.iloc[:,-2:]], axis=1)

#test
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        data_new.iloc[:,[0,1,2,3,5,6]],data_new.iloc[:,4:5], 
        test_size = 0.33,random_state = 0)

#linear regression
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train)

y_pred = linear_reg.predict(x_test)

#backward elimination
import statsmodels.formula.api as sm

#x = np.append(arr = np.ones((14,1)).astype(int), 
#              values=sonveri.iloc[:,[0,1,2,3,5,6]], axis=1)
x_list = data_new.iloc[:,[0,1,2,3,5,6]].values
r_ols = sm.OLS(endog = data_new.iloc[:,4:5], exog=x_list).fit()
print(r_ols.summary())

x_list = data_new.iloc[:,[0,1,2,3,6]].values
r_ols = sm.OLS(endog = data_new.iloc[:,4:5], exog=x_list).fit()
print(r_ols.summary())

x_train = x_train.iloc[:,[0,1,2,3,5]]
x_test = x_test.iloc[:,[0,1,2,3,5]]




