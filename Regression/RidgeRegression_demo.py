import numpy as np
import matplotlib.pyplot as plt

#岭回归
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx +np.eye(np.shape(xMat)[1])*lam
    if np.linalg.det(denom) == 0.0:
        print('THis matrix is singular, cannot do inverse')
    ws =  denom.I* (xMat.T*yMat)
    return ws

def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i-10))
        wMat[i,:] = ws.T
    return wMat

def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = np.mean(inMat,axis=0)   #calc mean then subtract it off
    inVar = np.var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar #减去均值，再除以标准差
    return inMat

X,Y = loadDataSet("abalone.txt")
ridgeWeights = ridgeTest(X,Y)
print(ridgeWeights.shape)
import matplotlib.pyplot as plt
n = ridgeWeights.shape[1]#特征数
numTestPts = 30 ###
for i in range(n):
    plt.plot(np.arange(numTestPts)-10, ridgeWeights[:,i], label = "W(%s)"%i)
plt.legend(loc="upper right")
#$\lambda\ $ 注意这里的空格符
plt.title(r"岭回归 回归系数随$\lambda\ $变化规律""\n基于数据集'abalone.txt'", fontsize=16)
plt.xlabel(r"ln($\lambda\ $)")
plt.grid(ls="--",lw =1)
plt.show()

xMat_normilized = regularize(np.mat(X))
rssErrorList=[]
for i in range(numTestPts):
    Y_predict = xMat_normilized*ridgeWeights[i].reshape(n,1) +np.mean(np.mat(Y).T,axis =0)
    RSS = rssError(np.array(Y).T, np.array(Y_predict))
    rssErrorList.append(RSS)
plt.plot(np.arange(numTestPts)-10, rssErrorList)
plt.xlabel(r"ln($\lambda\ $)")
plt.ylabel(r"RSS")
plt.show()
for i in range(20,25):
    Y_predict = xMat_normilized*ridgeWeights[i].reshape(n,1) +np.mean(np.mat(Y).T,axis =0)
    RSS = rssError(np.array(Y).T, np.array(Y_predict))
    #rssErrorList.append(RSS)
    plt.plot(np.array(Y)[1000:1100],label= "Y_True(train set)")
    plt.plot(Y_predict[1000:1100], label="Y_Predict")
    plt.title("ln("r"$\lambda\ $"")=%d, RSS =%.1f "%(i-10, RSS))
    plt.legend()
    plt.show()
