from mxnet import nd
from mxnet.gluon import nn

class MLP(nn.Bolock):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='Relu')
        self.output = nn.Dense(10)
        def forward(self, x):
            return self.output(self.hidden(x))

X = nd.random.uniform(shape=(2,20))
net = MLP()
net.initialize()
net(X)

class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)

    def add(self, block):
        self._children[block.name] = block

    def forward(self, x):
        for block in self._children.values():
            x = block(x)
        return x

#构造复杂的模型

class FancyMLP(nn.Block):
    def __intit__(self, **kwargs):
        super(FancyMLP, self).__intit__(**kwargs)
        self.rand_weight = self.params.get_contant('rand_weight', nd.random.uniform(shape = (20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, x):
        x =self.dense(x)
        x = nd.relu(nd.dot(x, self.rand_weight.data())+1)
        x = self.dense(x)
        while x.norm().asscalar() > 1:
            x /= 2
        if x.norm().asscalar() < 0.8:
            x *= 10
        return x.sum()

class NestNLP(nn.Block):
    def __init__(self, **kwargs):
        super(NestNLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'), nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, x):
        return self.dense(self.net(x))

