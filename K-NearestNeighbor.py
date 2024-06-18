from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#load data
df = pd.read_csv('teleCust1000t.csv')
#convert df to np
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values
#labels (our categories)
y = df['custcat'].values

#normalize data: Data Standardization gives the data zero mean and unit variance
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

#Train and test sets
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

#classify with the training set
#I start with k=4
k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
#make predictions with the test set
y_predicted = neigh.predict(X_test)

#calculate the classification accuracy
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, y_predicted))

#the accuracy is far for 1, meaning that the classification is not very accurate
#I now use k=3
k=3
#Train Model and Predict on train set
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
#predict on the test set
y_predicted = neigh.predict(X_test)
#accuracy
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, y_predicted))

#the accuracy is still far for 1, meaning that the classification is not very accurate
#I now use k=5
k=5
#Train Model and Predict on train set
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
#predict on the test set
y_predicted = neigh.predict(X_test)
#accuracy
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, y_predicted))

#the best turns out to be k=3
