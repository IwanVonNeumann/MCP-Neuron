import matplotlib.pyplot as plt

from data_access import load_iris_data


# y = df.iloc[0:100, 4].values
# y = np.where(y == "Iris-setosa", -1, 1)

def two_classes_plot():
    iris_df = load_iris_data()

    columns = ["sepal length", "petal length", "class"]

    set_X = iris_df[columns].loc[iris_df["class"] == "Iris-setosa"]
    ver_X = iris_df[columns].loc[iris_df["class"] == "Iris-versicolor"]

    plt.scatter(set_X["sepal length"], set_X["petal length"], color="red", marker="o", label="setosa")
    plt.scatter(ver_X["sepal length"], ver_X["petal length"], color="blue", marker="x", label="versicolor")
    plt.xlabel("petal length")
    plt.ylabel("sepal length")
    plt.legend(loc="upper left")
    plt.show()


def three_classes_plot():
    iris_df = load_iris_data()

    columns = ["sepal length", "petal length", "class"]

    set_X = iris_df[columns].loc[iris_df["class"] == "Iris-setosa"]
    ver_X = iris_df[columns].loc[iris_df["class"] == "Iris-versicolor"]
    vir_X = iris_df[columns].loc[iris_df["class"] == "Iris-virginica"]

    plt.scatter(set_X["sepal length"], set_X["petal length"], color="blue", marker="x", label="setosa")
    plt.scatter(ver_X["sepal length"], ver_X["petal length"], color="red", marker="o", label="versicolor")
    plt.scatter(vir_X["sepal length"], vir_X["petal length"], color="limegreen", marker="^", label="virginica")
    plt.xlabel("petal length")
    plt.ylabel("sepal length")
    plt.legend(loc="upper left")
    plt.show()
