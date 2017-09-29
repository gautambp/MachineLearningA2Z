# -*- coding: utf-8 -*-

# read dataset
import pandas as pd
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3,4]].values

# Utilize the elbow method to pick optimal # of clusters
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
wcss = []
for i in range(1, 11):
    model = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    model.fit(x)
    wcss.append(model.inertia_)
plt.plot(range(1, 11), wcss)
plt.show()

# As per the elbow chart, # of clusters = 5 seems to be optimal
# build model and predict cluster for each data points
no_cluster = 5
model = KMeans(n_clusters=no_cluster, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_pred = model.fit_predict(x)

# visualize clusters and data-points in it
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(0, no_cluster):
    label = 'Cluster' + str(i)
    plt.scatter(x[y_pred == i, 0], x[y_pred == i, 1], s=100, c=colors[i], label=label)
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], s=300, c='yellow', label='Centroids')
plt.legend()
plt.show()
