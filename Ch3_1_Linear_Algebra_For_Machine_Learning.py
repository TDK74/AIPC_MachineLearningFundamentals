import numpy as np
import matplotlib.pyplot as plt

# Optimal Salary Problem
x = np.linspace(0, 50, 100)
y1 = 30 * x + 1000 # salary #1 equation
y2 = 50 * x + 100 # salary #2 equation

plt.plot(x, y1, c='orange')
plt.plot(x, y2, c='blue')
