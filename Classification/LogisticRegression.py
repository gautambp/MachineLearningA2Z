# -*- coding: utf-8 -*-

# read dataset
import pandas as pd
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

# split dataset into train and test
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/4, random_state=0)

# scale all the features
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# create and train the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)
model.fit(x_train, y_train)

# predict on the test data and learn about performance using confusion matrix
y_pred = model.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# define a scatter plot visualization func
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def visualize(x_set, y_set):
    cmap = ListedColormap(('red', 'blue'))
    X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = cmap)
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):    
        plt.scatter(x_set[y_set == j,0], x_set[y_set == j,1], c=cmap(j))
    plt.show()

# visualize the train data
visualize(x_train, y_train)

# visualize the test data
visualize(x_test, y_test)

# visualize the prediction data
visualize(x_test, y_pred)
