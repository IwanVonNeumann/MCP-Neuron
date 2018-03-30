from data_access.iris import load_iris_data
from plots.raw_data import classes_scatter_plot, two_classes_scatter_plot

iris_df = load_iris_data()

classes_scatter_plot(iris_df)
# two_classes_scatter_plot(iris_df, ["Iris-setosa", "Iris-versicolor"])
