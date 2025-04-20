import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])
model = LinearRegression().fit(x, y)

plt.scatter(x, y, color='blue')
plt.plot(x, model.predict(x), color='red')
plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
