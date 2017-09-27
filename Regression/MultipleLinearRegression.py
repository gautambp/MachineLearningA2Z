# -*- coding: utf-8 -*-

# read dataset
import pandas as pd
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# take care of variable like - state using LabelEncoder and then separate out in multiple columns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
stateLabelEnc = LabelEncoder()
x[:, 3] = stateLabelEnc.fit_transform(x[:, 3])
stateOneEnc = OneHotEncoder(categorical_features=[3])
x = stateOneEnc.fit_transform(x).toarray()
# remove one state column as it is redundant
x = x[:,1:]

# split dataset into train and test
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Create the model and fit it to the train data
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

# predict values for test dataset
y_pred = model.predict(x_test)

# Build new model with backward elimination process and see if it's optimal
import statsmodels.formula.api as sm
import numpy as np
x = np.append(arr=np.ones((50,1)).astype(int), values=x, axis=1)
# choose the variables you want to feed to the model
# initially we choose all variables
x_opt = x[:, [0,1,2,3,4,5]]
ols_model = sm.OLS(endog=y, exog=x_opt).fit()
print('p-values : ', ols_model.pvalues)
# p value is highest for third variable, so we eliminate and try again
x_opt = x[:, [0,1,3,4,5]]
ols_model = sm.OLS(endog=y, exog=x_opt).fit()
print('p-values : ', ols_model.pvalues)
# now p value for second variable is max, so we eliminate it
x_opt = x[:, [0,3,4,5]]
ols_model = sm.OLS(endog=y, exog=x_opt).fit()
print('p-values : ', ols_model.pvalues)
# keep doing the same thing..
x_opt = x[:, [0,3,5]]
ols_model = sm.OLS(endog=y, exog=x_opt).fit()
print('p-values : ', ols_model.pvalues)

x_opt = x[:, [0,3]]
ols_model = sm.OLS(endog=y, exog=x_opt).fit()
print('p-values : ', ols_model.pvalues)

# we were able to eliminate 4 variables with threshold of 5% of p-value
# variables 0 & 3 (columns) are highest predictor of output value

