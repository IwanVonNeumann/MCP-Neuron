from data_access.iris import load_iris_data
from neurons.perceptron import Perceptron
from plots.learning_results import plot_2d_decision_boundary, plot_error_history
from utils.data_transformation import binarize_set_ver_label

iris_df = load_iris_data()

setosa_versicolor = ["Iris-setosa", "Iris-versicolor"]

set_ver_df = iris_df[iris_df["class"].isin(setosa_versicolor)].copy()
set_ver_df["binary class"] = set_ver_df.apply(binarize_set_ver_label, axis=1)

predictor_columns = ["sepal length", "petal length"]

train_X = set_ver_df[predictor_columns].values
train_y = set_ver_df["binary class"].values

perceptron = Perceptron(eta=0.1, n_iter=10)
perceptron.fit(train_X, train_y)

# plot_error_history(perceptron.errors_history_)
plot_2d_decision_boundary(set_ver_df, perceptron)
