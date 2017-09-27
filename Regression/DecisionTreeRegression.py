# -*- coding: utf-8 -*-

# read dataset
import pandas as pd
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# build and train Decision Tree model
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=0)
model.fit(x, y)

# Visualize the data
import matplotlib.pyplot as plt
import numpy as np
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='black')
plt.plot(x_grid, model.predict(x_grid), color='red')
plt.show()
