import numpy as np

class LinearRegression:

    def __init__(self, n_iterarions=1000, rate=0.0001):
        self.n_iterarions = n_iterarions
        self.rate = rate
        self.weights = None
        self.bias = 0
        self.costs = None

    def fit(self, X, y):
        self.costs = np.empty(self.n_iterarions)
        y = y.reshape(-1, 1)
        self.n_samples, self.n_features = X.shape
        self.weights = np.zeros(self.n_features)

        for i in range(self.n_iterarions):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / self.n_samples) * np.dot(X.T, y_pred - y)
            db = (1 / self.n_samples) * np.sum(y_pred - y)
            self.weights -= self.rate * dw
            self.bias -= self.rate * db
            self.costs[i] = self._cost(y_pred, y)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def _cost(self, y1, y2):
        return (2 / self.n_samples) * np.sum((y1 - y2)**2)

    def score(self, y_pred, y):
        return 1 - np.mean((y_pred - y)**2)/np.mean((y - y.mean())**2)
    