# -*- coding: utf-8 -*-

# read dataset
import pandas as pd
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

# convert dataset to list of list (apriori model expects list of list)
row, column = dataset.shape
x = []
for i in range(0, row):
    x.append([str(dataset.values[i, j]) for j in range(0, column)])

# train the model
from apyori import apriori
rules = apriori(x, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)
results = list(rules)

