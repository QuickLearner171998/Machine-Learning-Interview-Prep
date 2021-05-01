# -*- coding: utf-8 -*-
"""
https://towardsdatascience.com/logistic-regression-from-scratch-69db4f587e17

Step1 z = w1.x1 + w2.x2...
Step2 ypred = sigmoid(z)
Step3 Loss =  -y*ln(1-ypred) - (1-y)*ln(1-ypred)
Step 4 GD compute gradients dL/dW = X*(ypred - y)
Step 5 Update Weights w = w - alpha*dL/dW
Step 6 Repeat above steps

"""


import pandas as pd
import numpy as np
from numpy.random import rand

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class LogisticRegression:

    def sigmoid(self, z):
        return 1/(1+(np.e)**(-z))

    def cost_function(self, ypred, y, weights, m):
        return (-1/m)*(y*np.log(ypred) + (1-y)*np.log(1-ypred))

    def fit(self, X, y, epochs=1, lr=0.05):
        loss = []
        weights = rand(X.shape[1])
        N = len(X)

        for epoch in range(epochs):
            z = np.dot(X, weights)
            ypred = self.sigmoid(z)
            l = sum(self.cost_function(ypred, y, weights, N))
            print("Epoch : {} || Loss : {}".format(epoch, l))
            loss.append(l)
            # compute gradients
            grad = np.dot(X.T, ypred - y)/N
            # update weights
            weights = weights - lr*(grad)

        self.weights = weights
        self.loss = loss

    def predict(self, X):
        z = np.dot(X, self.weights)
        ypred = self.sigmoid(z)
        return [1 if i > 0.5 else 0 for i in ypred]


# GEt data
X = load_breast_cancer()['data']
y = load_breast_cancer()['target']
feature_names = load_breast_cancer()['feature_names']
dataset = pd.DataFrame(np.concatenate((X, y[:,None]), axis=1), columns = 
np.append(feature_names, 'Target'))

scaler = MinMaxScaler(feature_range=(-1,1))
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train, epochs=500, lr=0.5)
y_pred = log_reg.predict(X_test)

print(classification_report(y_test, y_pred))
print('-'*55)
print('Confusion Matrix\n')
print(confusion_matrix(y_test, y_pred))

