#！/usr/bin/env python
# -*- coding:utf-8 -*-

from mxnet import autograd, nd
from mxnet.gluon import nn
from PIL import Image
import pylab
import matplotlib.pyplot as plt
import os
import numpy as np

def corr2d(X, K):
    h, w = K.shape
    Y = nd.zeros((X.shape[0]-h+1, X.shape[1]+w-1))
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i+h, j: j+w]*K).sum()
    return Y

#2d_convolutional_layer
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x,self.weight.data()) + self.bias.data()

#图像中的物体边缘检测
#通过数据学习核数组

conv2d = nn.Conv2D(1,kernel_size=(1, 2))
conv2d.initialize()

#通过python模拟实现的图像卷积操作
#模拟 sobel算子, prewitt算子

def conv(image, kernel):
    height, width = image.shape  # 获取图像的维度
    h, w = kernel.shape  # 卷积核的维度

    # 经过卷积操作后得到的新的图像的尺寸
    new_h = height - h + 1
    new_w = width - w + 1
    # 对新的图像矩阵进行初始化
    new_image = np.zeros((new_h, new_w), dtype=np.float)

    # 进行卷积操作，矩阵对应元素值相乘
    for i in range(new_w):
        for j in range(new_h):
            new_image[i, j] = np.sum(image[i:i + h, j:j + w] * kernel)  # 矩阵元素相乘累加

    # 去掉矩阵乘法后的小于0的和大于255的原值，重置为0和255
    # 用clip函数处理矩阵的元素，使元素值处于（0，255）之间
    new_image = new_image.clip(0, 255)

    # 将新图像各元素的值四舍五入，然后转成8位无符号整型
    new_image = np.rint(new_image).astype('uint8')
    return new_image


if __name__ == "__main__":

    # 读取图像信息，并转换为numpy下的数组
    image = Image.open("图片.jpg", 'r')
    output_path = "./outputPic/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    a = np.array(image)

    # sobel 算子
    sobel_x = np.array(([-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]))
    sobel_y = np.array(([-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]))
    sobel = np.array(([-1, -1, 0],
                      [-1, 0, 1],
                      [0, 1, 1]))

    # prewitt各个方向上的算子
    prewitt_x = np.array(([-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]))
    prewitt_y = np.array(([-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]))
    prewitt = np.array(([-2, -1, 0],
                        [-1, 0, 1],
                        [0, 1, 2]))

    # 拉普拉斯算子
    laplacian = np.array(([0, -1, 0],
                          [-1, 4, -1],
                          [0, -1, 0]))
    laplacian_2 = np.array(([-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]))

    kernel_list = ("sobel_x", "sobel_y", "sobel", "prewitt_x", "prewitt_y", "prewitt", "laplacian", "laplacian_2")

    print("Gridient detection\n")
    for w in kernel_list:
        print("starting %s....." % w)
        print("kernel:\n")
        print("R\n")
        R = conv(a[:, :, 0], eval(w))
        print("G\n")
        G = conv(a[:, :, 1], eval(w))
        print("B\n")
        B = conv(a[:, :, 2], eval(w))

        I = np.stack((R, G, B), axis=2)  # 合并三个通道的结果
        Image.fromarray(I).save("%s//bigger-%s.jpg" % (output_path, w))

#多通道卷积
import d2lzh as d2l
from mxnet import nd

def corr2d_multi_in(X, K):
    #首先沿着X，K的通道维遍历
    return nd.add_n(*[d2l.corr2d(x,k) for x,k in zip(X,K)])

#多输出通道
def corr2d_multi_in_out(X, K):
    return nd.stack(*[corr2d_multi_in(X,k) for k in K])

#1x1卷积层
# 1x1 卷积层被当作保持高和宽维度形状不变的全连接层使用
def corr2d_multi_in_out_1x1(X,K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h*w))
    K = K.reshape((c_o, c_i))
    Y = nd.dot(K, X)
    return Y.reshape((c_o, h, w))

