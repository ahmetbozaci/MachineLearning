# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 15:16:13 2018

@author: Ahmet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("salary.csv")

x = data.iloc[:,1:2].values
y = data.iloc[:,2:].values

#%%Linear regression
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(x,y)

#%%polynomial regression 2.degree
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x)
linear_reg2 = LinearRegression()
linear_reg2.fit(x_poly,y)

#%%polynomial regression 4.degree
poly_reg3 = PolynomialFeatures(degree = 4)
x_poly3 = poly_reg3.fit_transform(x)
linear_reg3 = LinearRegression()
linear_reg3.fit(x_poly3,y)

#%% visuzalition
plt.scatter(x,y,color = "blue")
plt.plot(x,linear_reg.predict(x), color = "red", label = "linear")
plt.legend()
plt.show()

plt.scatter(x,y, color = "blue")
plt.plot(x,linear_reg2.predict(x_poly), color = "green", label = "2.polinom")
plt.legend()
plt.show()

plt.scatter(x,y, color = "blue")
plt.plot(x,linear_reg3.predict(x_poly3), color = "black", label = "4.polinom")
plt.legend()
plt.show()

from sklearn.metrics import r2_score
print("polynomial R2:", r2_score(y,linear_reg2.predict(x_poly)))

import statsmodels.api as sm
ols = sm.OLS(endog=y, exog=x_poly).fit()
print(ols.summary())
