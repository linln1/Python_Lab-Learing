import d2lzh as d2l
from mxnet.gluon import data as gdata, loss as gloss, nn
from mxnet import autograd, init, nd
import mxnet.gluon
import numpy as np
import pandas as pd

#读取数据
train_data = pd.read_csv('../data/kaggle_house_pred_train.csv')
test_data = pd.read_csv('../data/kaggle_house_pred_test.csv')
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

#预处理数据
numeric_features = all_features.dtype[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply( lambda x: (x-x.mean())/(x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)

#dummy_na = True
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape

n_train = train_data.shape[0]
train_features = nd.array(all_features[:n_train].values)
test_features = nd.array(all_features[n_train:].values)
train_labels = nd.array(train_data.SalePrice.values).reshape((-1, 1))

loss = gloss.L2loss()
def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net

def log_rmse(net, features, labels):
    clipped_pred = nd.clip(net(features), 1, float('inf'))
    rmse = nd.sqrt(2 * loss(clipped_pred.log(), labels.log()).mean())
    return rmse.asscalar()

def train(net, train_features, train_labels, test_features, test_lables, num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate':learning_rate, 'wd':weight_decay})
    for epoch in range(num_epochs):
        for X,y in train_iter:
            with autograd.record():
