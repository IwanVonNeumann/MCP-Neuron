import numpy as np


class Adaline(object):
    """ADAptive LInear NEuron classifier.
    Parameters
    ------------
    learning_rate : float
    Learning rate (between 0.0 and 1.0)
    n_iter : int
    Passes over the training dataset.
    Attributes
    -----------
    w_ : 1d-array
    Weights after fitting.
    cost_history_ : list
    Number of misclassifications in every epoch.
    """

    def __init__(self, learning_rate=0.01, n_iter=50):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        """ Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
        Training vectors,
        where n_samples is the number of samples and
        n_features is the number of features.
        y : array-like, shape = [n_samples]
        Target values.
        Returns
        -------
        self : object
        """

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
        return self

    def net_input(self, X):
        """Calculate net input"""
        return X.dot(self.w_) + self.b_

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)
