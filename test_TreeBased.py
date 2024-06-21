from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTree import DecisionTree
from RandomForest import RandomForest

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

classifier = DecisionTree(max_depth=14)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

print(np.sum(y_test == predictions)/len(y_test))

classifier2 = RandomForest(n_trees=20)
classifier2.fit(X_train, y_train)
predictions2 = classifier2.predict(X_test)

print(np.sum(y_test == predictions2)/len(y_test))