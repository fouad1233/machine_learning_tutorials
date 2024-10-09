import numpy as np
import matplotlib.pyplot as plt
from ml import regression

#make a parabola with some noise
x = np.linspace(-10, 10, 100)
r = 2 * x**2 + 3 * x + 5 + np.random.normal(0, 10, x.shape)



w = regression(x, r, 3)
print(w)

#plot the parabola
x_new = np.linspace(-10, 10, 100)
y_pred = sum([w[i] * x_new**i for i in range(3)])
plt.plot(x_new, y_pred, 'r')
plt.scatter(x, r)
plt.show()


