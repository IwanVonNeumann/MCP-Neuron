import numpy as np


class Perceptron(object):
    """Perceptron classifier.
    w_: Weights after fitting, 1d-array
    errors_: Number of misclassifications in every epoch, list
    """

    def __init__(self, learning_rate=0.01, n_iter=10):
        self.learning_rate = learning_rate  # float in (0.0, 1.0)
        self.n_iter = n_iter  # Passes over the training dataset, int

    def fit(self, X, Y):
        """Fit training data.
        X: Training vectors, {array-like}, shape = [n_samples, n_features], where
        n_samples is the number of samples and
        n_features is the number of features.
        y: Target values, array-like, shape = [n_samples]
        """

        n, m = X.shape

        self.w_ = np.zeros(m)
        self.b_ = 0

        self.errors_history_ = []

        for _ in range(self.n_iter):
            errors = 0
            for x, y in zip(X, Y):
                y_pred = self.predict(x)
                self.w_ += self.learning_rate * (y - y_pred) * x
                self.b_ += self.learning_rate * (y - y_pred)
                errors += int(y != y_pred)
            self.errors_history_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return X.dot(self.w_) + self.b_

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
