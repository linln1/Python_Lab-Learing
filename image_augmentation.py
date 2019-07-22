#! /usr/bin/env python
# -*- coding:utf-8 -*-

import d2lzh as d2l
import numpy as np
import mxnet as mx
from mxnet import nd,autograd,gluon, image, init
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils
import sys
import time

def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols* scale, num_rows* scale)
    _, axes = d2l.plt.subplot(num_rows, num_cols, figsize = figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j].asnumpy())
            axes[i][j].axes.get_xaixs().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes

#为了方便观察图像增广的效果,定义一个辅助函数apply,对增光函数aug展示所有结果
def apply(img, aug, num_rows = 2, num_cols= 4, scale = 1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)
#图像反转
#apply(img, gdata.vision.transforms.RandomFlipLeftRight() )
#图像裁剪
# shape_aug = gdata.vision.transforms.RandomResizedCrop( (200, 200), scale=(0.1,1), ratio=(0.5,2))
# apply(img, shape_aug)

#变化颜色：亮度(Brightness)，对比度(Contrast)，饱和度(Saturation)，色调(Hue)
#apply(img, gdata.vision.transforms.RandomBrightness(0.5))  1-0.5 ~ 1+0.5
#apply(img, gdata.vision.transforms.RandomHue(0.5))
#color_aug = gdata.vision.transforms.RandomColorJitter( bightness = 0.5, contrast=0.5, saturation = 0.5, hue = 0.5)
#apply(img, color_aug)

#叠加多个图像增光方法
# augs = gdata.vision.transforms.Compose([ gdata.vision.transforms.RandomFlipLeftRight(), color_aug, shape_aug])

#CIFAR-10 图像增广训练模型
# show_images(gdata.vision.CIFAR10(train=True)[0:32][0], 4, 8, scale = 0.8)

flip_aug = gdata.vision.transforms.Compose([gdata.vision.transforms.RandomFlipLeftRight(),
                                            gdata.vision.transforms.ToTensor()])

no_aug = gdata.vision.transforms.Compose([gdata.vision.transforms.ToTensor()])

num_workers = 0 if sys.platform.startswith('win32') else 4
def load_cifar10(is_train, augs, batch_size):
    return gdata.DataLoader(
        gdata.vision.CIFAR10(train=is_train).transform_first(augs),
        batch_size = batch_size, shuffle=is_train, num_workers=num_workers
    )

#使用多GPU训练模型
def try_all_gpus():
    ctxes = []
    try:
        for i in range(16):
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx= ctx)
            ctxes.append(ctx)
    except mx.base.MXNetError:
        pass
    if not ctxes:
        ctxes = [mx.cpu()]
    return ctxes

def _get_batch(batch, ctx):
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx), features.shape[0])

def evaluate_accuracy(data_iter, net, ctx = [mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc_sum, n = nd.array([0]), 0
    for batch in data_iter:
        features, labels, _ = _get_batch(batch, ctx)
        for X,y in zip(features, labels):
            y = y.astype('float32')
            acc_sum += (net(X).argmax(axis = 1) == y ).sum().copyto(mx.cpu())
            n += y.size
        acc_sum.wait_to_read()
    return acc_sum.asscalar() / n

def train(train_iter, test_iter, net, loss, trainer, ctx, num_epoches):
    print('training on ', ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(num_epoches):
        train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
        for i, batch in enumerate(train_iter):
            Xs, ys, batch_size = _get_batch(batch, ctx)
            ls = []
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
            for l in ls:
                l.backward()
            train.step(batch_size)
            train_l_sum += sum([l.sum().asscalar() for l in ls])
            n += sum([l.size for l in ls])
            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar() for y_hat, y in zip(y_hats, ys) ])
            m += sum([y.size for y in ys])
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f ,test acc %.3f, time %.1f sec' % 
              (epoch + 1, train_l_sum/n, train_acc_sum / m, test_acc, time.time() - start))
        
def train_with_data_aug(train_augs, test_augs, lr = 0.001):
    batch_size ,ctx, net = 256, try_all_gpus(), d2l.resnet18(10)
    net.initialize(ctx = ctx, init= init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate':lr})
    loss = gloss.SoftmaxCrossEntropyLoss()
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    train(train_iter, test_iter, net, loss, trainer, ctx, num_epoches=10)
    
    
    
