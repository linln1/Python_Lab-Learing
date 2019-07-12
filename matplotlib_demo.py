#！ /usr/bin/env python
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

#柱状图
# plt.style.use('ggplot')
# customers = ['ABC','DEF','GHI','JKL','MNO']
# customers_index = range(len(customers))
# sales_amounts = [127, 90, 201, 111, 232]
# fig = plt.figure()
# ax1 = fig.add_subplot(1,1,1)
# ax1.bar(customers_index, sales_amounts, align='center', color='darkblue')
# ax1.xaxis.set_ticks_position('bottom')
# ax1.yaxis.set_ticks_position('left')
# plt.xticks(customers_index, customers, rotation=0, fontsize='small')
# plt.xlabel('Customer Name')
# plt.ylabel('Sale Amount')
# plt.title('Sale Amount per Customer')
# plt.savefig('bar_plot.png', dpi=400, bbox_inches = 'tight')
# plt.show()

#直方图

# plt.style.use('ggplot')
# mu1 ,mu2, sigma = 100, 130, 15
# x1 = mu1 + sigma*np.random.randn(10000)
# x2 = mu2 + sigma*np.random.randn(10000)
# fig = plt.figure()
# ax1 = fig.add_subplot(1,1,1)
# n ,bins, patches = ax1.hist(x1, bins=50, normed=False, color='darkgreen')
# n, bins, patches = ax1.hist(x2, bins=50, normed=False, color='orange', alpha=0.5)
# ax1.xaxis.set_ticks_position('bottom')
# ax1.yaxis.set_ticks_position('left')
# plt.xlabel('Bins')
# plt.ylabel('Number of Values in Bin')
# fig.suptitle('Histograms', fontsize = 14, fontweight = 'bold')
# ax1.set_title('Two Frequency Distributions')
# plt.savefig('histogram.png', dpi=400, bbox_inches = 'tight')
# plt.show()

#折线图
# from numpy.random import randn
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
# plot_data1 = randn(50).cumsum()
# plot_data2 = randn(50).cumsum()
# plot_data3 = randn(50).cumsum()
# plot_data4 = randn(50).cumsum()
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(plot_data1, marker=r'o', color=u'blue', linestyle='-', label='Blue Solid')
# ax1.plot(plot_data2, marker=r'+', color=u'red' , linestyle='--', label='Red Dashed')
# ax1.plot(plot_data3, marker=r'*', color=u'green', linestyle='-.', label='Green Dash Dot')
# ax1.plot(plot_data4, marker=r's', color=u'orange', linestyle=':', label='Orange Dotted')
# ax1.xaxis.set_ticks_position('bottom')
# ax1.yaxis.set_ticks_position('left')
# ax1.set_title('Line Plots: Markers, Colors, and Linestyles')
# plt.xlabel('Draw')
# plt.ylabel('Random Number')
# plt.legend(loc = 'best')
# plt.savefig('line_plot.png', dpi=400, bbox_inches = 'tight')
# plt.show()

#散点图
plt.style.use('ggplot')
x = np.arange(start=1., stop=15., step=1.)
y_linear = x + 5.0*np.random.randn(14)
y_quadratic = x**2 + 10.0*np.random.randn(14)
fn_linear = np.poly1d(np.polyfit(x, y_linear, deg = 1))
fn_quadratic = np.poly1d(np.polyfit(x, y_quadratic, deg = 2))
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.plot(x, y_linear, 'bo', x, y_quadratic, 'go', x, fn_linear(x), 'b-', x, fn_quadratic(x), 'g-', linewidth=2)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')
ax1.set_title('Scatter Plots Regression Lines')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.xlim((min(x)-1., max(x)+1))
plt.ylim((min(y_quadratic)-10., max(y_quadratic)+10.))
plt.savefig('scatter_plot.png', dpi=400, bbox_inches='tight')
plt.show()

#箱线图

