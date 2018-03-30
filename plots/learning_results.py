import matplotlib.pyplot as plt


def plot_error_history(errors_history):
    epoch_numbers = range(1, len(errors_history) + 1)
    plt.plot(epoch_numbers, errors_history, marker="o")
    plt.title("Errors history")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.show()


# TODO refactor to find classes
def plot_2d_decision_boundary(iris_df, classifier):
    set_df = iris_df[iris_df["class"] == "Iris-setosa"]
    ver_df = iris_df[iris_df["class"] == "Iris-versicolor"]

    min_x, max_x, min_y, max_y, des_y1, des_y2 = decision_boundary(iris_df, classifier)
    des_x = [min_x, max_x]
    des_y = [des_y1, des_y2]

    plt.plot(des_x, des_y, color="navy", linewidth=2)
    plt.fill_between(des_x, min_y, des_y, color="blue", alpha=0.25)
    plt.fill_between(des_x, des_y, max_y, color="red", alpha=0.25)

    plt.scatter(set_df["sepal length"], set_df["petal length"], color="blue", marker="x", label="setosa")
    plt.scatter(ver_df["sepal length"], ver_df["petal length"], color="red", marker="o", label="versicolor")

    plt.title("2 classes decision boundary")
    plt.xlabel("sepal length")
    plt.ylabel("petal length")
    plt.legend(loc="upper left")
    plt.show()


def decision_boundary(iris_df, classifier):
    setosa_versicolor = ["Iris-setosa", "Iris-versicolor"]
    set_ver_df = iris_df[iris_df["class"].isin(setosa_versicolor)]

    b = 0.5

    min_x = set_ver_df["sepal length"].min() - b
    max_x = set_ver_df["sepal length"].max() + b

    min_y = set_ver_df["petal length"].min() - b
    max_y = set_ver_df["petal length"].max() + b

    w = classifier.w_
    des_y1 = - (w[0] + w[1] * min_x) / w[2]
    des_y2 = - (w[0] + w[1] * max_x) / w[2]

    return min_x, max_x, min_y, max_y, des_y1, des_y2
