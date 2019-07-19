#! usr/bin/env python
# -*- coding:utf-8 -*-

import d2lzh as d2l
from mxnet import gluon, nd, init
from mxnet.gluon import nn

class Residual(nn.Block):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, strides=strides, padding=1)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1, strides=strides)

        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, x):
        Y = nd.relu(self.bn1(self.conv1(x)))
        Y = nd.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(x)
        return nd.relu(Y+X)


#ResNet与GoogLeNet 不同之处在于ResNet每个卷积层后增加的批量归一化层

net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3), nn.BatchNorm(), nn.Activation('relu'), nn.MaxPool2D(pool_size=3, strides=2, padding=1))


#DenseNet 稠密块(dense block) 和过渡层(transition layer)

def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'), nn.Conv2D(num_channels, kernel_size=3, padding=1))
    return blk

class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = nd.concat(X, Y, dim=1)
        return X

def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'), nn.Conv2D(num_channels, kernel_size=1), nn.AvgPool2D(pool_size=2, strides=2))
    return blk


