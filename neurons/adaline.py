import time

import numpy as np


class Adaline:

    def __init__(self, learning_rate=0.01, n_iter=50):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        start = time.time()

        n, m = X.shape

        self.w_ = np.zeros(m)
        self.b_ = 0

        self.cost_history_ = []

        for _ in range(self.n_iter):
            a = self.activation(X)
            self.w_ += self.learning_rate * X.T.dot(y - a)
            self.b_ += self.learning_rate * np.sum(y - a)

            cost = np.sum((y - a) ** 2) / 2
            self.cost_history_.append(cost)

        end = time.time()
        self.learning_time_ = end - start

        return self

    def net_input(self, X):
        return X.dot(self.w_) + self.b_

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

    def cost(self, X, y):
        return (1 / 2) * ((y - self.activation(X)) ** 2).sum()
