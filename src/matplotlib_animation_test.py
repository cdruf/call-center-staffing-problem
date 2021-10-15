from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

fig = plt.figure()

x = [1, 2, 3, 4, 5]
y = [5, 7, 2, 5, 3]

data = np.column_stack([np.linspace(0, yi, 50) for yi in y])

rects = plt.bar(x, data[0], color='c')
line, = plt.plot(x, data[0], color='r')
plt.ylim(0, max(y))


def animate(i):
    for rect, yi in zip(rects, data[i]):
        rect.set_height(yi)
    line.set_data(x, data[i])
    return rects, line


anim = animation.FuncAnimation(fig, animate, frames=len(data), interval=40)
plt.show()
