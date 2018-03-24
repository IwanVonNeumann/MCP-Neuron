import matplotlib.pyplot as plt

from data_access.iris import load_iris_data


# y = df.iloc[0:100, 4].values
# y = np.where(y == "Iris-setosa", -1, 1)

def two_classes_plot():
    iris_df = load_iris_data()

    set_df = iris_df.loc[iris_df["class"] == "Iris-setosa"]
    ver_df = iris_df.loc[iris_df["class"] == "Iris-versicolor"]

    plt.scatter(set_df["sepal length"], set_df["petal length"], color="red", marker="o", label="setosa")
    plt.scatter(ver_df["sepal length"], ver_df["petal length"], color="blue", marker="x", label="versicolor")
    plt.xlabel("petal length")
    plt.ylabel("sepal length")
    plt.legend(loc="upper left")
    plt.show()


def three_classes_plot():
    iris_df = load_iris_data()

    set_df = iris_df.loc[iris_df["class"] == "Iris-setosa"]
    ver_df = iris_df.loc[iris_df["class"] == "Iris-versicolor"]
    vir_df = iris_df.loc[iris_df["class"] == "Iris-virginica"]

    plt.scatter(set_df["sepal length"], set_df["petal length"], color="blue", marker="x", label="setosa")
    plt.scatter(ver_df["sepal length"], ver_df["petal length"], color="red", marker="o", label="versicolor")
    plt.scatter(vir_df["sepal length"], vir_df["petal length"], color="lime", marker="^", label="virginica")
    plt.xlabel("petal length")
    plt.ylabel("sepal length")
    plt.legend(loc="upper left")
    plt.show()
