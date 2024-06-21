from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTree import DecisionTree

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

classifier = DecisionTree()
classifier.fit(X_train, y_train)
predictions = classifier(X_test)

print(np.sum(y_test == predictions)/len(y_test))