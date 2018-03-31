from data_access.iris import load_iris_data
from neurons.adaline_sgd import AdalineSGD
from utils.data_transformation import standardize_columns, binarize_class_labels
from plots.learning_results import plot_2d_decision_boundary, plot_error_history

iris_df = load_iris_data()

setosa_versicolor = ["Iris-setosa", "Iris-versicolor"]
predictor_columns = ["sepal length", "petal length"]

set_ver_df = iris_df[iris_df["class"].isin(setosa_versicolor)].copy()
set_ver_df = binarize_class_labels(set_ver_df)
set_ver_df_s = standardize_columns(set_ver_df, predictor_columns)

train_X = set_ver_df_s[predictor_columns].values
train_y = set_ver_df_s["binary class"].values

adaline = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
adaline.fit(train_X, train_y)

error_plot_settings = {
    "title": "Adaline Stochastic Gradient Descent",
    "xlabel": "Epochs",
    "ylabel": "Average Cost"
}
# plot_error_history(adaline.cost_history_, error_plot_settings)
plot_2d_decision_boundary(set_ver_df_s, adaline, support_vectors=True)
