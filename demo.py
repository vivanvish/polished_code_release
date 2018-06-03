"""
    Example file to demonstrate how to use the KernelSVM classifer.
    I will be using the MNIST data for this.
"""

from KernelSVM import KernelSVM

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.datasets import load_digits
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

print('......Simulated data.......')
print('Creating the data and splitting into train and test sets.')
X,y = make_classification(n_samples=500,n_features=10,n_informative=7,n_classes=5)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2, random_state=42)

print('Preprocessing the data.')
scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(' Using RBF Kernel.')
print('Training the classiffer with different combinations of hyperparameters.')

combinations = np.array(np.meshgrid([0.01,1,4,8,16,32],[0.01,0.1,0.5])).T.reshape(-1,2)
for params in combinations:
    clf = KernelSVM(lamda=params[0],kernel='rbf',sigma=params[1])
    clf.mysvm(X_train,y_train)
    preds = clf.predict(X_train,X_test)
    print('Misclassification Error with lamda=',params[0],', sigma=',params[1],' : ',1-accuracy_score(y_test,preds))



print('......MNIST data.......')
print('Loading the data and splitting into train and test sets.')
X, y = load_digits(return_X_y =True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2, random_state=42)

print('Preprocessing the data.')
scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(' Using Polynomial Kernel.')
print('Training the classiffer with different combinations of hyperparameters.')

combinations = np.array(np.meshgrid([0.01,1,4,8,16,32],[3,10,15])).T.reshape(-1,2)
for params in combinations:
    clf = KernelSVM(lamda=params[0],kernel='polynomial',p=params[1])
    clf.mysvm(X_train,y_train)
    preds = clf.predict(X_train,X_test)
    print('Misclassification Error with lamda=',params[0],', p=',params[1],' : ',1-accuracy_score(y_test,preds))


print(' Using RBF Kernel.')
print('Training the classiffer with different combinations of hyperparameters.')

combinations = np.array(np.meshgrid([0.01,1,4,8,16,32],[0.01,0.1,0.5])).T.reshape(-1,2)
for params in combinations:
    clf = KernelSVM(lamda=params[0],kernel='rbf',sigma=params[1])
    clf.mysvm(X_train,y_train)
    preds = clf.predict(X_train,X_test)
    print('Misclassification Error with lamda=',params[0],', sigma=',params[1],' : ',1-accuracy_score(y_test,preds))

