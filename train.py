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

classifier = kNN(k=10)
classifier.fit(X_train, y_train)
preds = classifier.predict(X_test)


for id, pred in enumerate(preds):
    print(f"prediction : {pred} | actual : {y_test[id]} | correct : {'Yes' if pred == y_test[id] else 'No'}")
    
print(f"accuracy : {sum(preds == y_test)/len(y_test)}")