# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 02:32:18 2018

@author: Ahmet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("salary.csv")

x = data.iloc[:,1:2].values
y = data.iloc[:,2:].values

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=0)
tree_reg.fit(x,y)

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)

plt.scatter(x,y, color="red")
plt.plot(x_,tree_reg.predict(x_), color="blue")
plt.show()

from sklearn.metrics import r2_score
print("Decisiontree R2:", r2_score(y,tree_reg.predict(x)))
