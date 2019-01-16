import numpy as np

from math import ceil
from sklearn.utils import shuffle as sklearn_shuffle


class AdalineMGD:

    def __init__(self, learning_rate=0.01, n_iter=50):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y, batch_size=16):
        n, m = X.shape

        self.w_ = np.zeros(m)
        self.b_ = 0

        self.cost_history_ = []

        for _ in range(self.n_iter):
            X_batches, y_batches = self.split_into_batches(X, y, size=batch_size, shuffle=True)
            for X_b, y_b in zip(X_batches, y_batches):
                a_b = self.activation(X_b)
                self.w_ += self.learning_rate * X_b.T.dot(y_b - a_b)
                self.b_ += self.learning_rate * np.sum(y_b - a_b)

            a = self.activation(X)
            cost = np.sum((y - a) ** 2) / 2
            self.cost_history_.append(cost)
        return self

    def split_into_batches(self, X, y, size=16, shuffle=True):
        if shuffle:
            X, y = sklearn_shuffle(X, y, random_state=0)

        n, m = X.shape
        n_batches = ceil(n / size)

        X_b = [X[i * size: (i + 1) * size, :] for i in range(n_batches)]
        y_b = [y[i * size: (i + 1) * size] for i in range(n_batches)]

        return X_b, y_b

    def net_input(self, X):
        return X.dot(self.w_) + self.b_

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)
