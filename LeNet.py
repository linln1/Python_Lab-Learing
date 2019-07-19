#! /usr/bin/env python
# -*- coding:utf-8 -*-

#LeNet 模块分为卷积层块 和 最大池化层

#卷积层块的基本单位是卷积层后接最大池化层： 卷积层用来识别图像里的空间模式 如线条和局部物体 之后最大池化层则用来降低卷积层对位置的敏感性
#卷积层块有两个这样的基本单位重复堆叠而成

#每个卷积层都使用5x5的窗口并在输出上使用sigmoid激活函数第一个卷积层的输入通道数位6， 第二个卷积层输出通道数增加到16.这hi因为第二个卷积层比第一个卷积层的输入的高和宽都要笑，所以增加输出通道使两个卷积层参数尺寸类似

import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
import time

net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, activation='sigmoid'), nn.MaxPool2D(pool_size=2, strides=2), nn.Conv2D(channels=16, kernel_size=5,activation='sigmoid'), nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120,activation='sigmoid'),
        nn.Dense(84, activation='sigmoid'),
        nn.Dense(10)
        )
X = nd.random.uniform(shape(1, 1, 28, 28))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)

#训练模型
batch_size = 256
train_iter ,test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx

def evalutee_accuracy(data_iter, net, ctx):
    acc_sum, n = nd.array([0], ctx=ctx), 0
    for X,y in data_iter:
        X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y ).sum()
        n += y.size
    return acc_sum.asscalar() / n

def train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs):
    print('training on', ctx)
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X,y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test_acc,', 'time %.1f sec' % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, time.time() - start))

lr, num_epoch = 0.9, 5
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)

