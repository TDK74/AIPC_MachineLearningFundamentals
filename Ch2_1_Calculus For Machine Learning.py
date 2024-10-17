import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 3, 100)
y = ((x**2) * -1) + 3*x - 1

plt.plot(x, y)
