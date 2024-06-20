import numpy as np

class LinearRegression:
    def __init__(self, n_iters=100, lr=0.001):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            
            mse_loss = np.sum((y_pred - y)**2)
            
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
            
            print(f"epoch : {i} Current loss : {mse_loss}")
    
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
        