#lasso(least absolute shrikage and select operator)

def lasso_regression(X, y ,lambd=0.2, threshold=0.1):
    rss = lambda X, y, w:(y - X*w).T*(y - X*w)
    m, n = X.shape
    w = np.mat(np.zeros((n,1)))
    r = rss(X,y,w)
    nither = itertools.count(1)
    for it in nither:
        for k in range(n):
            z_k = (X[:,k].T*X[:,k])[0,0]
            p_k = 0
            for i in range(m):
                p_k += X[i,k]*(y[i,0] - sum([X[i,j]*w[j,0] for j in range(n)]))
            if p_k < -lambd/2:
                w_k = (p_k + lambd/2)/z_k
            elif p_k < -lambd/2:
                w_k = (p_k - lambd/2)/z_k
            else:
                w_k =0
            w[k, 0] = w_k
        r_prime = rss(X,y,w)
        delta = abs(r_prime - r)[0,0]
        r = r_prime
        print('Iteration: {}, delta = []'.format(it,delta))
        if delta<threshold:
            break
    return w

#lasso 回归系数轨迹
def lasso_traj(X,y , ntest=30):
    _, n =X.shape
    ws = np.zeros((ntest,n))
    for i in range(ntest):
        w = ridgeRegres(X, y, lambd=np.exp(i-10))
        ws[i,:] = w.T
    return ws

# if '__main__' == __name__:
#     X, y = loadDataSet('abalone.txt')
#     X, y = standarize(X), standarize(y)
#     w = lasso_regression(X, y , lambd=10)
#     y_prime = X*w
#     corrcoef = get_corrcoef(np.array(y.reshape(1,-1)),np.array(y_prime.reshape(1,-1)))
#
# ntest = 30
# ws = lasso_traj(X, y,ntest)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# lambdas = [i-10 for i in range(ntest)]
# plt.show()
