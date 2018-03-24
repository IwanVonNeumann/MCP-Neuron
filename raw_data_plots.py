from data_access.iris import load_iris_data
from plots.raw_data import two_classes_plot, three_classes_plot

iris_df = load_iris_data()

# two_classes_plot(iris_df)
three_classes_plot(iris_df)
