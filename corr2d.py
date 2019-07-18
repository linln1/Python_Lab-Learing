#！/usr/bin/env python
# -*- coding:utf-8 -*-

from mxnet import autograd, nd
from mxnet.gluon import nn

def corr2d(X, K):
    h, w = K.shape
    Y = nd.zeros((X.shape[0]-h+1, X.shape[1]+w-1))
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i+h, j: j+w]*K).sum()
    return Y
