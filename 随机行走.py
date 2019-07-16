import random
import matplotlib.pyplot as plt

position = 0
walk = [position]
for i in range(500):
        step = 1 if random.randint(0,1) else -1
        position += step
        walk.append(position)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(walk, marker=r'', color=u'black')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_title('Random walk with +1/-1 steps')
plt.show()
