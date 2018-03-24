import matplotlib.pyplot as plt


def two_classes_plot(iris_df):
    set_df = iris_df[iris_df["class"] == "Iris-setosa"]
    ver_df = iris_df[iris_df["class"] == "Iris-versicolor"]

    plt.scatter(set_df["sepal length"], set_df["petal length"], color="blue", marker="x", label="setosa")
    plt.scatter(ver_df["sepal length"], ver_df["petal length"], color="red", marker="o", label="versicolor")

    plt.title("2 classes of Iris flowers")
    plt.xlabel("sepal length")
    plt.ylabel("petal length")
    plt.legend(loc="upper left")
    plt.show()


def three_classes_plot(iris_df):
    set_df = iris_df[iris_df["class"] == "Iris-setosa"]
    ver_df = iris_df[iris_df["class"] == "Iris-versicolor"]
    vir_df = iris_df[iris_df["class"] == "Iris-virginica"]

    plt.scatter(set_df["sepal length"], set_df["petal length"], color="blue", marker="x", label="setosa")
    plt.scatter(ver_df["sepal length"], ver_df["petal length"], color="red", marker="o", label="versicolor")
    plt.scatter(vir_df["sepal length"], vir_df["petal length"], color="lime", marker="^", label="virginica")

    plt.title("3 classes of Iris flowers")
    plt.xlabel("sepal length")
    plt.ylabel("petal length")
    plt.legend(loc="upper left")
    plt.show()
