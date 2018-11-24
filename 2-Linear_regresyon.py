# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:45:36 2018

@author: Ahmet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("sales.csv")
months = data[["Months"]]
sales = data[["Sales"]]

#train and test
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        months, sales, test_size = 0.33, random_state = 0 )

#feature scaling
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

x1_train = scale.fit_transform(x_train)
x1_test = scale.fit_transform(x_test)
y1_train = scale.fit_transform(y_train)
y1_test = scale.fit_transform(y_test)

#modeling
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train)
y_pred = linear_reg.predict(months)

linear_reg.predict(x_test)

#visuzalition

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train, y_train)
plt.plot(x_test, linear_reg.predict(x_test))

from sklearn.metrics import r2_score
print("linear r2:", r2_score(sales,y_pred))





