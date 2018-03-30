from data_access.iris import load_iris_data
from plots.raw_data import classes_scatter_plot

all_classes_df = load_iris_data()

# classes_scatter_plot(all_classes_df)

setosa_versicolor = ["Iris-setosa", "Iris-versicolor"]
set_ver_df = all_classes_df[all_classes_df["class"].isin(setosa_versicolor)].copy()
classes_scatter_plot(set_ver_df)
