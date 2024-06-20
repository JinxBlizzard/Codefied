import numpy as np

def sigmoid(x):
    x_new = np.array(x, dtype=np.float64)
    return 1/(1 + np.exp(-x))

def log_loss(y, preds):
    return np.dot(y, np.log(preds)) + np.dot((1 - y), np.log(1 - preds)) 
    

class LogisticRegression():
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iters):
            linear_predictions = np.dot(X, self.weights) + self.bias
            prob_predictions = sigmoid(linear_predictions)
            
            dw = (1/n_samples) * np.dot(X.T, (prob_predictions - y))
            db = (1/n_samples) * np.sum(prob_predictions - y)
            
            self.weights = self.weights - dw * self.lr
            self.bias = self.bias - db * self.lr
            
            loss = log_loss(y, prob_predictions)
            
            print(f"epoch : {i} current_loss : {loss}")
            
    
    def predict(self, X):
        linear_predictions = np.dot(X, self.weights) + self.bias
        prob_predictions = sigmoid(linear_predictions)
        preds = [0 if ele<=0.5 else 1 for ele in prob_predictions]
        
        return preds