# -*- coding: utf-8 -*-

# read dataset
import pandas as pd
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# split dataset into train and test
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

# Create the model and fit it to the train data
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

# predict values for test dataset
y_pred = model.predict(x_test)

# visualize train, test, against actual and predicted values
import matplotlib.pyplot as plt
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, model.predict(x_train), color='red')
plt.scatter(x_test, y_test, color='blue')
plt.plot(x_test, y_pred, color='blue')
plt.show()
