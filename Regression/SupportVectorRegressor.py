# -*- coding: utf-8 -*-

# read dataset
import pandas as pd
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# since SVR model library does not perform feature scaling, we've to manually do it
from sklearn.preprocessing import StandardScaler
x_scale = StandardScaler()
y_scale = StandardScaler()
x = x_scale.fit_transform(x)
y = y_scale.fit_transform(y)

# build and train SVR model
from sklearn.svm import SVR
model = SVR(kernel='rbf')
model.fit(x, y)

# Visualize the data
import matplotlib.pyplot as plt
plt.scatter(x, y, color='black')
plt.plot(x, model.predict(x), color='red')
plt.show()
