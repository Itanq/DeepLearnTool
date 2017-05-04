#coding:utf-8


import os
import numpy as np
import matplotlib.pyplot as plt


data1 = np.loadtxt("../../Dataset/logistic_data1.txt", delimiter=',')

X = data1[:,0:2]
Y = data1[:,2]

pos = np.where(Y==1)
neg = np.where(Y==0)

plt.scatter(X[pos,0], X[pos,1], marker='o', c='b')
plt.scatter(X[neg,0], X[neg,1], marker='x', c='r')
plt.legend(['Fail', 'Pass'])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def cost(theta, x, y):



