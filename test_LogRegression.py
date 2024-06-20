import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression

ds = datasets .load_breast_cancer()
X, y = ds.data, ds.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)


def pltt(y_preds=None):
    fig = plt.figure(figsize=(8,6))
    plt.scatter(X[:,0], y, color='b', marker='o', s=30)
    if y_preds is not None:
        plt.scatter(X_test[:,0], y_preds, color='r', marker='x', s=30)
        plt.scatter(X_test[:,0], y_test, color='b', marker='o', s=30)
    plt.show()
    
def pltt_pred(y_preds=None):
    if y_preds is None:
        return
    fig = plt.figure(figsize=(8,6))
    plt.scatter(X_test[:,0], y_preds, color='r', marker='x', s=30)
    plt.scatter(X_test[:,0], y_test, color='b', marker='o', s=30)
    plt.show()
        
print_var = input("create visual for the dataset? ")

if print_var.lower() == "yes":
    pltt()
    
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_preds = classifier.predict(X_test)

pltt_pred(y_preds=y_preds)

print(np.sum(y_preds == y_test)/len(y_preds))