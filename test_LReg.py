import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

X, y = datasets.make_regression(n_samples=100, n_features=3, noise=20, random_state=7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=71)

def plttr(X_test=None, y_test=None):
    fig = plt.figure(figsize=(8,6))
    plt.scatter(X[:,0], y, color='b', marker='o', s=30)
    if X_test is not None and y_test is not None:
        plt.scatter(X_test[:,0], y_test, color='r', marker='o', s=30)
    plt.show()
    
print_var = input("Print Dataset? ")

if print_var.lower() == "yes":
    plttr()

reg = LinearRegression(n_iters=1000, lr=0.01)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

plttr(X_test, predictions)