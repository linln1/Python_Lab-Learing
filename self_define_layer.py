#！ /usr/bin/env python
# -*- coding:utf -*-

from mxnet import gluon,nd
from mxnet.gluon import nn

# 不含模型参数的自定义层
class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x-x.mean()

#韩模型参数的自定义层
params = gluon.ParameterDict()
params.get('param2', shape=(2, 3))

class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)



