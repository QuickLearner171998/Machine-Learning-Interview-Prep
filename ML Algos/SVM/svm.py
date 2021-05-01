"""https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47

Step1 Compute ypred = wx + b
Step2 Loss = (lambda)*|W|**2 + max(0, 1-ypred)
Step3 GD Gradients dL/dw = 2*lambda*W if y*ypred >=1 else 2*lambda*W - y*x
Step4 Update weights w = w- alpha*dL/dw
Step5 Repeat Above steps

Prediction wx+b >=1 ypred = 1, wx+b<=-1 ypred = -1
"""
import numpy as np
import pandas as pd


class SVM:
    def __init__(self, lr=0.01, lambda_param = 0.01, epochs = 100):
        self.lr = lr
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        y_ = np.where(y<=0,-1, 1) # if y<=0 replace with -1 else replace with 1
        n_features = X.shape[1]
        # INIT weights and b
        self.w = np.zeros(n_features)
        self.b = 0
        # TRAIN LOOP
        loss = []
        for _ in range(self.epochs):
            for i, x in enumerate(X):
                y_pred = np.dot(self.w, x) + self.b
                loss.append(self.lambda_param*np.square(np.linalg.norm(self.w)) + max(0, 1-y_pred))
                print(loss[-1])
                # Compute gradients
                if y_[i]*y_pred >=1:
                    self.w -= self.lr*(2*self.lambda_param*self.w)
                else:
                    self.w -= self.lr*(2*self.lambda_param*self.w - np.dot(x,y_[i]))
                    self.b -= self.lr*y_[i]
    
    def predict(self,X):
        y_pred = np.dot(X, self.w) + self.b
        return np.sign(y_pred)
