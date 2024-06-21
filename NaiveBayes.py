import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.u_classes = np.unique(y)
        n_classes = len(self.u_classes)
        
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)
        
        for idx, c in enumerate(self.u_classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0]/float(n_samples)
            
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []
        
        for idx, c in enumerate(self.u_classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)
            
        return self.u_classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numr = np.exp(-((x - mean) ** 2) / (2 * var))
        denr = np.sqrt(2 * np.pi * var)
        
        return numr/denr
    
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    
    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=7)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    
    classifier = NaiveBayes()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    print("accuracy:", np.sum(y_test == y_pred)/len(y_test))