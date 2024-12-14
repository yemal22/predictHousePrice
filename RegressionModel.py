import numpy as np
from sklearn.preprocessing import StandardScaler

class LinearRegression1:
    def __init__(self, n_iterations=100, rate=0.001):
        self.n_iterations = n_iterations
        self.rate = rate
        self.weights = None
        self.bias = 0
        self.costs = None
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.X_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(np.array(y).reshape(-1, 1))

        self.n_samples, self.n_features = X_scaled.shape

        self.weights = np.zeros((self.n_features, 1))
        self.bias = 0
        self.costs = np.zeros(self.n_iterations)

        for i in range(self.n_iterations):
            y_pred = np.dot(X_scaled, self.weights) + self.bias
            errors = y_pred - y_scaled

            dw = (1 / self.n_samples) * np.dot(X_scaled.T, errors)
            db = (1 / self.n_samples) * np.sum(errors)

            self.weights -= self.rate * dw
            self.bias -= self.rate * db

            self.costs[i] = np.mean(errors ** 2)
            
    def predict(self, X):
        X_scaled = self.X_scaler.transform(X)
        y_pred = np.dot(X_scaled, self.weights) + self.bias
        return self.y_scaler.inverse_transform(y_pred.reshape(-1, 1))

    def score(self, X, y):
        y_pred = self.predict(X)
        y = np.array(y).reshape(-1, 1)
        return 1 - np.mean((y_pred - y)**2) / np.mean((y - y.mean())**2)
