import numpy as np


class Adaline(object):
    """ADAptive LInear NEuron classifier.
    Parameters
    ------------
    eta : float
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

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
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
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_history_ = []
        X_ = self.__prepend_unit_column(X)

        for i in range(self.n_iter):
            errors = y - X_.dot(self.w_)
            dw = self.eta * X_.T.dot(errors)
            self.w_ += dw
            cost = (errors ** 2).sum() / 2.0
            self.cost_history_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        X_ = self.__prepend_unit_column(X)
        return np.dot(X_, self.w_)

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)

    @staticmethod
    def __prepend_unit_column(X):
        h, _ = X.shape
        unit_column = np.full((h, 1), 1)
        return np.append(unit_column, X, axis=1)
