#-*- encoding:utf-8

import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t'))-1
    dataMat = []
    labelmat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelmat.append(float(curLine[-1]))
    return dataMat, labelmat

def standRegress(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    xTx = xMat.T*xMat
    if np.linalg.det(xTx) == 0.0:
        print('This Matrix is singular , cannot do inverse')
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

def drawfitline(xMat, yMat):
    fig = plt.figure()
    ax = fig.add_subplot(11)
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
