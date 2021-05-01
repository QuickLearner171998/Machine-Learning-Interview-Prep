# -*- coding: utf-8 -*-
"""
https://machinelearningmastery.com/implement-simple-linear-regression-scratch-python/

Step1 Calculate x_mean, y_mean
Step2 Compute m,c as m = sum(x-x_mean)*sum(y-y_mean) / sum((x-x_mean)**2)
        c = y_mean - m*x_mean
Step3 Compute y_pred using the line
Step4 Calc Error and Rsquare

data
X Y
1 1
2 3
4 3
3 2
5 5
"""


import pandas as pd
import numpy as np


def calc_variance(X, x_mean):
    return np.sum(np.square(X - x_mean))


def calc_mean(X):
    return np.sum(X)/len(X)


def calc_covariance(X, x_mean, Y, y_mean):
    return np.sum((X-x_mean)*(Y-y_mean))


def calc_coef(x_mean, y_mean, cov, x_var):
    m = cov / x_var
    c = y_mean - m*x_mean
    return m, c


data = pd.read_csv('simple-data.csv', sep=' ')

X = data['X']
Y = data['Y']

x_mean = calc_mean(X)
y_mean = calc_mean(Y)

x_var = calc_variance(X, x_mean)
y_var = calc_variance(Y, y_mean)

# calculate Covarinace
cov = calc_covariance(X, x_mean, Y, y_mean)

# estimate coeff
m, c = calc_coef(x_mean, y_mean, cov, x_var)
