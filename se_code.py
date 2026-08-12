import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import sklearn.utils

X_height = np.array([[10.0], [7.0], [5.0], [8.0]])
Y_age = np.array([20, 14, 10, 16])

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
    X_height, Y_age, test_size=0.5
)

poly = make_pipeline(PolynomialFeatures(2), LinearRegression())
poly.fit(X_height, Y_age)

lin = linear_model.LinearRegression()
lin.fit(X_train, Y_train)

print(f"Train accuracy: {lin.score(X_train, Y_train) * 100:.2f}%")
print(f"Test accuracy: {lin.score(X_test, Y_test) * 100:.2f}%")
print(f"Poly Train accuracy: {poly.score(X_train, Y_train) * 100:.2f}%")
print(f"Poly Test accuracy: {poly.score(X_test, Y_test) * 100:.2f}%")

poly_test = poly.predict(np.array([[6.0], [9.0], [4.0]]))
print(poly_test)

plt.scatter(X_height, Y_age, color="red")
plt.scatter(X_test, lin.predict(X_test), color="green")
plt.scatter(X_test, Y_test, color="blue")
plt.plot(np.arange(0, 11), poly.predict(np.arange(0, 11).reshape(-1, 1)))

plt.savefig("plot.png")
plt.show()
