import numpy as np


class AdalineAnalytical:

    def __init__(self):
        self.coef_ = None

    def fit(self, X, y):
        X = self.__prepend_unit_column(X)
        A = X.T.dot(X)
        b = X.T.dot(y)
        self.coef_ = np.linalg.inv(A).dot(b)

        return self

    def net_input(self, X):
        X = self.__prepend_unit_column(X)
        return np.dot(X, self.coef_)

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

    @staticmethod
    def __prepend_unit_column(X):
        h, _ = X.shape
        unit_column = np.full((h, 1), 1)
        return np.append(unit_column, X, axis=1)

    def cost(self, X, y):
        return (1 / 2) * ((y - self.activation(X)) ** 2).sum()
