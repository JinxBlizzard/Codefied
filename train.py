import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
from kNN import kNN

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# # Plotting graph for visualization
# plt.figure()
# plt.scatter(X[:,2], X[:,3], c=y, cmap=cmap, edgecolor='k', s=20)
# plt.show()

classifier = kNN(k=5)
classifier.fit(X_train, y_train)
preds = classifier.predict(X_test)

print(preds[0][1][1])

# for id, pred in enumerate(preds):
#     len(pred)
#     print(pred, y_test[id])