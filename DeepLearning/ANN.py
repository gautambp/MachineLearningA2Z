# -*- coding: utf-8 -*-

# read dataset
import pandas as pd
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encode categorical data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le1 = LabelEncoder()
x[:,1] = le1.fit_transform(x[:,1])
le2 = LabelEncoder()
x[:,2] = le2.fit_transform(x[:,2])
ohe = OneHotEncoder(categorical_features=[1])
x = ohe.fit_transform(x).toarray()
x = x[:, 1:]

# split dataset into train and test
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/5, random_state=0)

# scale all the features
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# Build and train the model 
import keras
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
# Add input layers with first hidden layer
# 11 inputs as there are 11 features in the data
# 6 neurons in the hidden layer with relu activation function
model.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
# add another hidden layer
model.add(Dense(output_dim=6, init='uniform', activation='relu'))
# add output layer
# since it's binary classification, only one output
# also since it's binary output, we use sigmoid activation
model.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
# compile and finalize the model
# utilize adam as optimization function
# and use binary cross entropy as loss function (since it's binary output).. in case of linear
# generally we use RSME (squared error) based loss/cost func
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Now that model has been architected.. fit the data
model.fit(x_train, y_train, epochs=25)

# predict on the test data and learn about performance using confusion matrix
y_pred = model.predict(x_test)
# convert the prediction from 0 to 1 to boolean
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('accuracy = ', (cm[0][0] + cm[1][1])/y_test.size)

