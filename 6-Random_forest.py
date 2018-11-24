# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 16:44:04 2018

@author: Ahmet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("salary.csv")

x = data.iloc[:,1:2].values
y = data.iloc[:,2:].values

from sklearn.ensemble import RandomForestRegressor 
regressor = RandomForestRegressor(random_state=0, n_estimators=10)
regressor.fit(x,y)

plt.scatter(x,y, color="red")
plt.plot(x,regressor.predict(x), color="blue")
plt.show()

from sklearn.metrics import r2_score
print("Randomforest R2:", r2_score(y,regressor.predict(x)))