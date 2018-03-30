from data_access.iris import load_iris_data
from neurons.adaline import Adaline
from plots.learning_results import plot_2d_decision_boundary, plot_error_history
from utils.data_transformation import standardize_columns, binarize_class_labels

iris_df = load_iris_data()

versicolor_virginica = ["Iris-versicolor", "Iris-virginica"]
predictor_columns = ["sepal length", "petal length"]

ver_vir_df = iris_df[iris_df["class"].isin(versicolor_virginica)].copy()
ver_vir_df = binarize_class_labels(ver_vir_df)
ver_vir_df_s = standardize_columns(ver_vir_df, predictor_columns)

train_X = ver_vir_df_s[predictor_columns].values
train_y = ver_vir_df_s["binary class"].values

adaline = Adaline(n_iter=15, eta=0.01)
adaline.fit(train_X, train_y)

# plot_error_history(adaline.cost_history_)
plot_2d_decision_boundary(ver_vir_df_s, adaline, support_vectors=True)
