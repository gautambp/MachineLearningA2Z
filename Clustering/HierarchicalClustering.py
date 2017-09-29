# -*- coding: utf-8 -*-

# read dataset
import pandas as pd
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3,4]].values

# Utilize the dandograms to pick optimal # of clusters
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))
plt.show()

# Look for longest vertical line to the point where it intersects with horizontal line
# count total no of verical line before that intersection point.. in this case it is 5 (no of clusters)
no_cluster = 5

# prepare and train the model and predict cluster for each datapoint
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=no_cluster, affinity='euclidean', linkage='ward')
y_pred = model.fit_predict(x)

# visualize clusters and data-points in it
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(0, no_cluster):
    label = 'Cluster' + str(i)
    plt.scatter(x[y_pred == i, 0], x[y_pred == i, 1], s=100, c=colors[i], label=label)
#plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], s=300, c='yellow', label='Centroids')
plt.legend()
plt.show()
