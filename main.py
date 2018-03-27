import numpy as np
import matplotlib.pyplot as plt

from data_access.iris import load_iris_data
from neurons.adaline import Adaline
from neurons.perceptron import Perceptron
from plots.learning_results import plot_2d_decision_boundary, plot_error_history

iris_df = load_iris_data()


def binarize_label(row):
    binary_labels = {
        "Iris-setosa": -1,
        "Iris-versicolor": 1
    }
    return binary_labels[row["class"]]


setosa_versicolor = ["Iris-setosa", "Iris-versicolor"]

set_ver_df = iris_df[iris_df["class"].isin(setosa_versicolor)].copy()
set_ver_df["binary class"] = set_ver_df.apply(binarize_label, axis=1)

predictor_columns = ["sepal length", "petal length"]

train_X = set_ver_df[predictor_columns].values
train_y = set_ver_df["binary class"].values

# perceptron = Perceptron(eta=0.1, n_iter=10)
# perceptron.fit(train_X, train_y)

# plot_error_history(perceptron.errors_history_)

# plot_2d_decision_boundary(iris_df, perceptron)

# adaline = Adaline(eta=0.1, n_iter=10)
# adaline.fit(train_X, train_y)

# plot_error_history(adaline.cost_history_)

# plot_2d_decision_boundary(iris_df, adaline)

ada_1 = Adaline(n_iter=10, eta=0.01)
ada_1.fit(train_X, train_y)

for i, x in enumerate(ada_1.cost_history_):
    print(i, x)

ada_2 = Adaline(n_iter=10, eta=0.0001)
ada_2.fit(train_X, train_y)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

ax[0].plot(range(1, len(ada_1.cost_history_) + 1), np.log10(ada_1.cost_history_), marker='o')
# ax[0].plot(range(1, len(ada1.cost_history_) + 1), ada1.cost_history_, marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ax[1].plot(range(1, len(ada_2.cost_history_) + 1), ada_2.cost_history_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

plt.show()
