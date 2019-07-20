#! /usr/bin/env python
# -*- coding:utf-8 -*-

# import d2lzh as d2l
# import mxnet as mx
# import math
# from mxnet import nd
# import numpy as np
#
# # def gd(eta):
# #     x = 10
# #     results = [x]
# #     for i in range(10):
# #         x -= eta * 2 * x
# #         results.append(x)
# #     print('epoch 10, x:', x)
# #     return results
# #
# # def show_trace(res):
# #     n = max(abs(min(res)), abs(max(res)), 10)
# #     f_line = np.arange(-n, n, 0.1)
# #     d2l.set_figsize()
# #     d2l.plt.plot(f_line, [x * x for x in f_line])
# #     d2l.plt.plot(res, [x * x for x in res], '-o')
# #     d2l.plt.xlabel('x')
# #     d2l.plt.ylabel('f(x)')
# #     d2l.plt.show()
# #
# # if __name__ =='__main__':
# #     res = gd(1.1)
# #     show_trace(res)
#
# #learning_rate
#
# def train_2d(trainer):
#     x1, x2, s1, s2 = -5, -2, 0, 0
#     results = [(x1, x2)]
#     for i in range(20):
#         x1 ,x2 ,s1, s2 = trainer(x1,x2,s1,s2)
#         results.append((x1,x2))
#     print('epoch %d, x1%f, x2%f' % (i+1, x1, x2))
#     return results
#
# def show_trace_2d(f, results):
#     d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
#     x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, -1.0, 0.1))
#     d2l.plt.contour(x1, x2, f(x1,x2), colors='#1f77b4')
#     d2l.plt.xlabel('x1')
#     d2l.plt.ylabel('x2')
#     d2l.plt.show()
#
# eta = 0.1
# def f_2d(x1, x2):
#     return x1**2 + 2* x2**2
#
# def gd_2d(x1, x2, s1, s2):
#     return (x1-eta*2*x1, x2-eta*4*x2, 0 , 0)
#
# # if __name__ == '__main__':
# #     show_trace_2d(f_2d, train_2d(gd_2d))
#
# def sgd_2D(x1, x2, s1, s2):
#     return (x1-eta*(2*x1+np.random.normal(0.1)), x2-eta*(4*x2+np.random.normal(0.1)),0 ,0)
#
# if __name__ == '__main__':
#     show_trace_2d(f_2d, train_2d(sgd_2D))

#small_batch_stochastic_gradient_descent

#使用一个来自NASA的测试不同飞机机翼噪音的数据集来比较各个优化算法

import d2lzh as d2l
from mxnet import nd, autograd, init, gluon
from mxnet.gluon import nn, data as gdata, loss as gloss
import numpy as np
import time

def get_data_ch7():
    data = np.genfromtxt('../data/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis = 0)
    return nd.array(data[:1500, :-1]), nd.array(data[:1500, -1])

features, labels =get_data_ch7()
features.shape

def batch_sgd(params, states, hyperparams):
    for p in params:
        p[:] -= hyperparams['lr'] * p.grad

def train_ch7(trainer_fn, states, hyperparams, features, labels, batch_size =10, num_epochs = 2):
    net, loss = d2l.linreg, d2l.squared_loss
    w = nd.random.normal(scale=0.01, shape=(features.shape[1], 1))
    b = nd.zeros(1)
    w.attach_grad()
    b.attach_grad()

    def eval_loss():
        return loss(net(features, w, b),labels).mean().asscalar()

    ls = [eval_loss()]
    data_iter = gdata.DataLoader(
        gdata.ArrayDataset(features, labels), batch_size, shuffle=True
    )
    for _ in range(num_epochs):
        start = time.time()
        for batch_i ,(X,y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X, w, b), y).mean()
            l.backward()
            trainer_fn([w,b], states, hyperparams)
            if (batch_i + 1)*batch_size %100 == 0:
                ls.append(eval_loss())
        print('loss: %f,%f sec per epoch' % (ls[-1], time.time()-start))
        d2l.set_figsize()
        d2l.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
        d2l.plt.xlabel('epoch')
        d2l.plt.ylabel('loss')
        d2l.plt.show()

def train_sgd(lr, batch_size, num_epochs=2):
    train_ch7(batch_sgd, None, {'lr':lr}, features, labels, batch_size, num_epochs)


if __name__ == '__main__':
    train_sgd(1, 1500, 6)

#简洁实现
def train_gluon_ch7(trainer_name, trainer_hyperparams, features, labels, batch_size = 10, num_epochs=2):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    loss = gloss.L2Loss()
    
    def eval_loss():
        return loss(net(features), labels).mean().asscalar()
    ls = [eval_loss()]
    data_iter = gdata.DataLoader(
        gdata.ArrayDataset(features, labels), batch_size, shuffle=True
    )
    trainer = gluon.Trainer(net.collect_params(), trainer_name, trainer_hyperparams)
    for _ in range(num_epochs):
        start = time.time()
        for batch_i ,(X,y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
            if (batch_i+1 )* batch_size % 100 == 0:
                ls.append(eval_loss())
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    d2l.set_figsize()
    d2l.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('loss')
    d2l.plt.show()
    

    
