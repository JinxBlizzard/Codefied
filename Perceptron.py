import numpy as np

def unit_step_func(x):
    return np.where(x > 0, 1, 0)

class Perceptron:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.weights = np.random.randn(n_features)
        self.bias = np.random.randn()
        
        y_ = np.where(y > 0, 1, 0)
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                
                upd = self.lr * (y_[idx] - y_predicted)
                self.weights += upd*x_i
                self.bias += upd
                
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_preds = self.activation_func(linear_output)
        return y_preds
    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    
    X, y = datasets.make_blobs(n_features=2, n_samples=200, centers=2e, cluster_std=1.04, random_state=7)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    
    perp = Perceptron(lr=0.001, n_iters=1000)
    perp.fit(X_train, y_train)
    preds = perp.predict(X_train)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1,1,1)
    plt.scatter(X_train[:,0], X_train[:,1], marker='o', c=y_train)
    
    X01 = np.amin(X_train[:,0])
    X02 = np.amax(X_train[:,0])
    
    X11 = (-perp.weights[0] * X01 - perp.bias)/perp.weights[1]
    X12 = (-perp.weights[0] * X02 - perp.bias)/perp.weights[1]
    
    ax.plot([X01, X02], [X11, X12], "k")
    
    ymin = np.amin(X_train[:,1])
    ymax = np.amin(X_train[:,1])
    ax.set_ylim([ymin - 3, ymax + 3])
    
    plt.show()