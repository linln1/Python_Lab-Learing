#是一种贪心算法，效果和lasso差不多，但是好写地多
#一开始所有的权重都设为1，然后每一步所作的决策是对某个权重增加会减少一个很小的值，每一步都尽可能地减少误差
import numpy as np
import matplotlib.pyplot as plt


def stageWise(xArr, yArr, eps=0.01,numIt = 100):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    ymean = np.mean(yMat,0)
    yMat = yMat - ymean
    xMat = regularize(xMat)
    m,n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws= np.zeros((n,1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = np.inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE<lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat

