import matplotlib.pyplot as plt

from plots.utils import iris_plot_settings


def classes_scatter_plot(iris_df):
    class_labels = iris_df["class"].unique()
    split_df = {label: iris_df[iris_df["class"] == label] for label in class_labels}

    for label in class_labels:
        df = split_df[label]
        plot_settings = iris_plot_settings[label]
        plt.scatter(df["sepal length"], df["petal length"], **plot_settings)

    plt.title("Iris flowers classes")
    plt.xlabel("sepal length")
    plt.ylabel("petal length")
    plt.legend(loc="upper left")
    plt.show()
