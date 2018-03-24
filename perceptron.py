import numpy as np


class Perceptron(object):
    """Perceptron classifier.
    w_: Weights after fitting, 1d-array
    errors_: Number of misclassifications in every epoch, list
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta  # Learning rate, float in (0.0, 1.0)
        self.n_iter = n_iter  # Passes over the training dataset, int

    def fit(self, X, y):
        """Fit training data.
        X: Training vectors, {array-like}, shape = [n_samples, n_features], where
        n_samples is the number of samples and
        n_features is the number of features.
        y: Target values, array-like, shape = [n_samples]
        """

        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
