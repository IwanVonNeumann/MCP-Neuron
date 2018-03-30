from data_access.iris import load_iris_data
from utils.data_transformation import binarize_set_ver_label, standardize_columns
from neurons.adaline import Adaline
from plots.learning_results import plot_2d_decision_boundary

iris_df = load_iris_data()

setosa_versicolor = ["Iris-setosa", "Iris-versicolor"]
predictor_columns = ["sepal length", "petal length"]

set_ver_df = iris_df[iris_df["class"].isin(setosa_versicolor)].copy()
set_ver_df_s = standardize_columns(set_ver_df, predictor_columns)
set_ver_df_s["binary class"] = set_ver_df_s.apply(binarize_set_ver_label, axis=1)

train_X = set_ver_df_s[predictor_columns].values
train_y = set_ver_df_s["binary class"].values

adaline = Adaline(n_iter=15, eta=0.01)
adaline.fit(train_X, train_y)

# plot_error_history(adaline.cost_history_)
plot_2d_decision_boundary(set_ver_df_s, adaline, support_vectors=True)
