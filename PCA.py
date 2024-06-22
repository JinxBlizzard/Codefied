import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        
    def fit(self, X):
        self.mean = np.mean(X, axis = 0)
        X = X - self.mean
        
        cov = np.cov(X.T)
        
        eigenvectors, eigenvalues = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T
        
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        
        self.components = eigenvectors[:self.n_components]
        
    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn import datasets
    
    data = datasets.load_iris()
    X, y = data.data, data.target
    
    analyzer = PCA(3)
    analyzer.fit(X)
    X_projected = analyzer.transform(X)
    
    print(f"Shape of X: {X.shape}")
    print(f"Shape of transformed X: {X_projected.shape}")
    
    x1 = X_projected[:,0]
    x2 = X_projected[:,1]
    x3 = X_projected[:,2]
    
    plt.scatter(x1, x3, c=y, edgecolor=None, alpha=0.8, cmap="viridis")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar()
    plt.show()