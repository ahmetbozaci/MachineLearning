# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 02:00:09 2018

@author: Ahmet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score

data = pd.read_csv("salary_new.csv")

x = data.iloc[:,2:5].values
y = data.iloc[:,-1:].values

print(data.corr()) #
 
#%%Linear regression
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(x,y)

model1 = sm.OLS(linear_reg.predict(x),x)
print("Linear-Ols")
print(model1.fit().summary())

print("Linear R2 score")
print(r2_score(y,linear_reg.predict(x)))

#%%Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=4)
x_poly = poly.fit_transform(x)
linear_reg2 = LinearRegression()
linear_reg2.fit(x_poly,y)

model2 = sm.OLS(linear_reg2.predict(x_poly),x)
print("Polynom-Ols")
print(model2.fit().summary())

print("Polynomial R2 score")
print(r2_score(y,linear_reg2.predict(x_poly)))

#%%Support vektor regression 
from sklearn.preprocessing import StandardScaler
scale1 = StandardScaler()
x_scale = scale1.fit_transform(x)
scale2 = StandardScaler()
y_scale = scale2.fit_transform(y)

from sklearn.svm import SVR
svr_reg = SVR(kernel="rbf")
svr_reg.fit(x_scale,y_scale)

model3 = sm.OLS(svr_reg.predict(x_scale),x_scale)
print("Support Vektor OLS")
print(model3.fit().summary())

print("Suppor vektor R2")
print(r2_score(y_scale,svr_reg.predict(x_scale)))

#%%Decision Tree
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=0)
tree_reg.fit(x,y)

model4 = sm.OLS(tree_reg.predict(x),x)
print("Decision Tree OLS")
print(model4.fit().summary())

print("Decision Tree R2")
print(r2_score(y,tree_reg.predict(x)))

#%%Random Forest regressor
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(random_state=0, n_estimators=10)
forest_reg.fit(x,y)

model5 = sm.OLS(forest_reg.predict(x),x)
print("Random Forest OLS")
print(model5.fit().summary())

print("Random Forest R2")
print(r2_score(y,forest_reg.predict(x)))
