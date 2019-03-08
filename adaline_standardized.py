from data_access.iris import load_iris_data
from utils.data_transformation import standardize, binarize_class_labels
from neurons.adaline import Adaline
from plots.learning_results import plot_2d_decision_boundary, plot_error_history

iris_df = load_iris_data()

setosa_versicolor = ["Iris-setosa", "Iris-versicolor"]
predictor_columns = ["sepal length", "petal length"]

set_ver_df = iris_df[iris_df["class"].isin(setosa_versicolor)].copy()
set_ver_df = binarize_class_labels(set_ver_df)
set_ver_df_s = standardize(set_ver_df, predictor_columns)

train_X = set_ver_df_s[predictor_columns].values
train_y = set_ver_df_s["binary class"].values.reshape(-1, 1)

adaline = Adaline(n_iter=15, learning_rate=0.01)
adaline.fit(train_X, train_y)

error_plot_settings = {
    "title": "Adaline Gradient Descent",
    "xlabel": "Epochs",
    "ylabel": "Sum-squared-error"
}
# plot_error_history(adaline.cost_history_, error_plot_settings)
plot_2d_decision_boundary(set_ver_df_s, adaline, support_vectors=True)
