#Platt SMO( Sequential Minimal Optimization)

#SMO 算法是将大优化问题分解为多个小优化问题求解的
# SMO 算法的目标是求解出一系列的alpha 和 b
#SMO算法的工作原理是 每次循环中选择两个alpha进行优化处理 一旦找到一堆合适的alpha 就增大其中一个减小另一个  两个alpha 需要符合两个条件 alpha必须在间隔边界之外，两个alpha还没有进行过区间化处理或者不在边界上

#SMO 算法中的辅助函数

import numpy as np

def loadDataSet(filename):
    dataMat = [], labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append([float(lineArr[2])])
    return dataMat, labelMat

def selectJrand(i,m):
    j = i
    while(j==i):
        j = int(np.random.uniform(0,m))
    return j

def clipAlpha(aj, H,L):
    if aj>H:
        aj=H
    if aj<L:
        aj=L
    return aj

#SMO函数的伪代码大致如下: 
# # 创建一个alpha 向量并将其初始化为0向量
# 当迭代次数小于最大迭代次数时：
#     对数据集中的每个数据向量：
#         如果该数据向量可以被优化：
#             随机选择另一个数据向量
#             同时优化这两个向量
#             如果两个向量都不能被优化，退出内循环
# 如果所有向量都没有被优化，增加迭代数目，继续下一次循环

#Simplified SMO
def smoSimple(dataMatIn, classLables, C, toler, maxIter):
    dataMatrix = np.mat(dataMatIn); labelMatrix = np.mat(classLables).transpose()
    b = 0; m,n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m,1)))
    iter = 0
    while(iter<maxIter):
         alphaPairsChanged = 0
         for i in range(m):
             fXi = np.float(np.multiply(alphas, labelMatrix).T*(dataMatrix*dataMatrix[i,:].T)) + b
             Ei = fXi - np.float(labelMatrix[i])
             if ((labelMatrix[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMatrix[i]*Ei > toler) and (alphas[i] > 0)):
                 j = selectJrand(i,m)
                 fXj = np.float(np.multiply(alphas, labelMatrix).T*(dataMatrix*dataMatrix[j,:].T)) + b
                 Ej = fXj - np.float(labelMatrix[j])
                 alphaIold = alphas[i].copy()
                 alphaJold = alphas[j].copy()
                 if (labelMatrix[i] != labelMatrix[j]):
                     L = max(0, alphas[j] - alphas[i])
                     H = min(C, C+alphas[j] - alphas[i])
                 else:
                     L = max(0, alphas[j]+ alphas[i]-C)
                     H = min(C< alphas[i] + alphas[j])
                 if L==H: print('L=H') ;continue
                 eta = 2.0*dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T
                 if eta>=0 : print('eta>=0'); continue
                 alphas[j] -= labelMatrix[j]*(Ei-Ej)/eta
                 alphas[j] = clipAlpha(alphas[j], H, L)
                 if (abs(alphas[j] - alphaJold) < 0.00001):
                     print('j not moving enough'); continue
                 alphas[i] += labelMatrix[j]*labelMatrix[i]*(alphaJold - alphas[j])
                 b1 = b - Ei - labelMatrix[i]*(alphas[i] - alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMatrix[j]*(alphas[j] - alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                 b2 = b - Ej - labelMatrix[i]*(alphas[i] - alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMatrix[j]*(alphas[j] - alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                 if( 0 <alphas[i]) and (C>alphas[i]) : b = b1
                 elif (0<alphas[j]) and (C>alphas[j]) : b = b2
                 else: b = (b1+b2) /2.0
                 alphaPairsChanged += 1
                 print('iter : %d i:%d, paris changed %d' % (iter, i, alphaPairsChanged))
         if (alphaPairsChanged == 0): iter +=1
         else: iter = 0
         print('iteration number: %d' % iter)
    return b,alphas
             
                
# 为了得到支持向量的个数:
# np.shape(alphas[alphas>0])
# 为了得到那些数据点是支持向量，输入：
# for i in range(100):
#     if alphas[i]>0.0: print(dataArr[i], labelArr[i])

    
