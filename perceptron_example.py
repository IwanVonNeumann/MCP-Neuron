from data_access.iris import load_iris_data
from neurons.perceptron import Perceptron
from plots.learning_results import plot_2d_decision_boundary, plot_error_history
from utils.data_transformation import binarize_class_labels

iris_df = load_iris_data()

setosa_versicolor = ["Iris-setosa", "Iris-versicolor"]

set_ver_df = iris_df[iris_df["class"].isin(setosa_versicolor)].copy()
set_ver_df = binarize_class_labels(set_ver_df)

predictor_columns = ["sepal length", "petal length"]

train_X = set_ver_df[predictor_columns].values
train_y = set_ver_df["binary class"].values

perceptron = Perceptron(eta=0.1, n_iter=10)
perceptron.fit(train_X, train_y)

error_plot_settings = {
    "title": "Errors history",
    "xlabel": "Epochs",
    "ylabel": "Error"
}
# plot_error_history(perceptron.errors_history_, error_plot_settings)
plot_2d_decision_boundary(set_ver_df, perceptron)
