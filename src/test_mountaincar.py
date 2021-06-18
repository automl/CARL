import matplotlib.pyplot as plt
import numpy as np

x_min = -1.2
x_max = 0.6
X = np.linspace(x_min, x_max, 100)
Y = np.cos(3*X)
Y2 = np.sin(np.arctan(-3*np.sin(3*X)))

plt.plot(X, Y)
plt.plot(X, Y2)
plt.show()
