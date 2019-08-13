#coding = utf-8

import numpy as np
import tensorflow as tf
import os

def loadDataSet():
    datamat = [] ; labelmat = []
    fr = open("dataSet.txt")
    for line in fr.readlines():
        lineArr = line.strip().split()
        datamat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelmat.append(int(lineArr[2]))
    return datamat,labelmat

def sigmoid(x):
    return 1/(1+exp(-x))

def tanh(x):
    return (exp(x)-exp(-x))/(exp(x)+exp(-x))

def relu(x):
    return 0 if x<0 else x

def GradientDescent(datamat,classlabels):
    dataMatrix = mat(datamat)   
    labelMatrix = mat(classlabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001
    steps = 500
    w = ones((n,1))
    for i in range(steps):
        h=sigmoid(dataMatrix*w)
        error = labelMatirx - h
        w = w - alpha*dataMatrix.transpose()*error
    return w

def GradientAScent(datamat,classlabels):
    dataMatrix = mat(datamat)
    labelMatirx = mat(classlabels).transpose()
    m,n = shape(dataMatrix)
    steps = 500
    beta = 0.001
    w = ones((n,1))
    for i in range(beta):
         h = sigmoid(dataMatrix*w)
         error = labelMatrix - h
         w = w - beta*dataMatrix.transpose()*error
    return w


#画出数据边界

def plotBestFit(weight):
    import matplotlib.pyplot as plt
    weights = weight.getA()
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1=[]
    xcord2 = []; ycord2=[]
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, color='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, color='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stocGradAscent0(dataMatrix, classLables):
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLables[i] - h
        weights = weights + alpha*error*dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
    m,n = np.shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        for i in range(m):
            alpha = 4/(i+j+1.0) + 0.01
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[i]*weights))
            error = dataMatrix[i] - h
            weights = weights + alpha*error*dataMatrix[i]
    return weights

#这种改变alpha单调下降的方法同样用于模拟退火等算法
#第二点 通过随机取样来更新回归系数，这种方法可以减少周期性的波动
#第三点 这样的方法比以前收敛的更快

