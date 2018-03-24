import matplotlib.pyplot as plt


def plot_error_history(errors_history):
    epoch_numbers = range(1, len(errors_history) + 1)
    plt.plot(epoch_numbers, errors_history, marker="o")
    plt.title("Errors history")
    plt.xlabel("Epochs")
    plt.ylabel("Number of misclassifications")
    plt.show()


def plot_2d_decision_boundary(iris_df, classifier):
    set_df = iris_df[iris_df["class"] == "Iris-setosa"]
    ver_df = iris_df[iris_df["class"] == "Iris-versicolor"]

    min_x = min(set_df["sepal length"].min(), ver_df["sepal length"].min())
    max_x = max(set_df["sepal length"].max(), ver_df["sepal length"].max())

    min_y = min(set_df["petal length"].min(), ver_df["petal length"].min())
    max_y = max(set_df["petal length"].max(), ver_df["petal length"].max())

    w = classifier.w_
    des_y1 = - (w[0] + w[1] * min_x) / w[2]
    des_y2 = - (w[0] + w[1] * max_x) / w[2]

    des_x = [min_x - 0.5, max_x + 0.5]
    des_y = [des_y1, des_y2]

    plt.plot(des_x, des_y, color="navy", linewidth=2)
    plt.fill_between(des_x, min_y - 0.5, des_y, color="blue", alpha=0.25)
    plt.fill_between(des_x, des_y, max_y + 0.5, color="red", alpha=0.25)

    plt.scatter(set_df["sepal length"], set_df["petal length"], color="blue", marker="x", label="setosa")
    plt.scatter(ver_df["sepal length"], ver_df["petal length"], color="red", marker="o", label="versicolor")

    plt.title("2 classes of Iris flowers")
    plt.xlabel("sepal length")
    plt.ylabel("petal length")
    plt.legend(loc="upper left")
    plt.show()
