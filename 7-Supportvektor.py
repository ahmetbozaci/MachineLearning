# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 14:41:31 2018

@author: Ahmet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("salary.csv")

x = data.iloc[:,1:2].values
y = data.iloc[:,2:].values

from sklearn.preprocessing import StandardScaler

scale1 = StandardScaler()
x_scale = scale1.fit_transform(x)
scale2 = StandardScaler()
y_scale = scale2.fit_transform(y)

from sklearn.svm import SVR
svr_reg = SVR(kernel = "rbf")
svr_reg.fit(x_scale,y_scale)

plt.scatter(x_scale,y_scale, color="red")
plt.plot(x_scale,svr_reg.predict(x_scale), color="blue")

print(svr_reg.predict(11))

from sklearn.metrics import r2_score
print("Supporvektor R2:", r2_score(y_scale,svr_reg.predict(x_scale)))