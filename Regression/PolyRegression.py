# -*- coding: utf-8 -*-

# read dataset
import pandas as pd
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# split dataset into train and test
#from sklearn.cross_validation import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Evaulate data using linear regressor and poly regressor
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# now utilize 2nd degree (or higher) polynomial to evaluate data
from sklearn.preprocessing import PolynomialFeatures
# create new x that includes poly features
poly_feature = PolynomialFeatures(degree=3)
x_poly = poly_feature.fit_transform(x)
poly_feature.fit(x_poly, y)
# use the linear regressor with input that includes poly features
poly_reg = LinearRegression()
poly_reg.fit(x_poly, y)

# Visualize the data
import matplotlib.pyplot as plt
plt.scatter(x, y, color='black')
plt.plot(x, lin_reg.predict(x), color='red')
plt.plot(x, poly_reg.predict(poly_feature.fit_transform(x)), color='blue')
plt.show()
