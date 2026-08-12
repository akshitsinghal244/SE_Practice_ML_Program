"""Linear and polynomial regression demonstration."""

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


x_height = np.array([[10.0], [7.0], [5.0], [8.0]])
y_age = np.array([20, 14, 10, 16])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x_height, y_age, test_size=0.5
)

poly = make_pipeline(PolynomialFeatures(2), LinearRegression())
poly.fit(x_height, y_age)

lin = linear_model.LinearRegression()
lin.fit(x_train, y_train)

print(f"Train accuracy: {lin.score(x_train, y_train) * 100:.2f}%")
print(f"Test accuracy: {lin.score(x_test, y_test) * 100:.2f}%")
print(f"Poly Train accuracy: {poly.score(x_train, y_train) * 100:.2f}%")
print(f"Poly Test accuracy: {poly.score(x_test, y_test) * 100:.2f}%")

poly_test = poly.predict(np.array([[6.0], [9.0], [4.0]]))
print(poly_test)

plt.scatter(x_height, y_age, color="red")
plt.scatter(x_test, lin.predict(x_test), color="green")
plt.scatter(x_test, y_test, color="blue")
plt.plot(
    np.arange(0, 11),
    poly.predict(np.arange(0, 11).reshape(-1, 1))
)

plt.savefig("plot.png")
plt.show()
