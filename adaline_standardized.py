from data_access.iris import load_iris_data
from data_utils import binarize_set_ver_label, standardize
from neurons.adaline import Adaline
from plots.learning_results import plot_error_history

iris_df = load_iris_data()

setosa_versicolor = ["Iris-setosa", "Iris-versicolor"]

set_ver_df = iris_df[iris_df["class"].isin(setosa_versicolor)].copy()
set_ver_df["binary class"] = set_ver_df.apply(binarize_set_ver_label, axis=1)

predictor_columns = ["sepal length", "petal length"]

train_X = standardize(set_ver_df[predictor_columns]).values
train_y = set_ver_df["binary class"].values

adaline = Adaline(n_iter=15, eta=0.01)
adaline.fit(train_X, train_y)

plot_error_history(adaline.cost_history_)
