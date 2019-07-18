#！ /usr/bin/env python
# -*- coding:utf-8 -*-

from mxnet import nd, init
from mxnet.gluon import nn

class MyInit(nn.Block):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)

#延后初始化 (defer initialization)
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'), nn.Dense(10))

net.initialize(init = MyInit())

X = nd.random.uniform(shape = (2,20))
Y = net(X)

# avoid defer initialization
# first_case re-initialization(cover old params)
net.initialize(init=MyInit(), force_reinit=True)
# second_case show num_inputs
net = nn.Sequential()
net.add(nn.Dense(256, in_units=20, activation='relu'))
net.add(nn.Dense(10, in_units=256))
net.initialize(init= MyInit())

