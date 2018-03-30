import matplotlib.pyplot as plt

from plots.utils import plot_settings


def plot_error_history(errors_history):
    epoch_numbers = range(1, len(errors_history) + 1)
    plt.plot(epoch_numbers, errors_history, marker="o")
    plt.title("Errors history")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.show()


def plot_2d_decision_boundary(iris_df, classifier, support_vectors=False):
    class_labels = iris_df["class"].unique()
    if len(class_labels) != 2:
        raise ValueError("DataFrame of two classes expected")

    split_df = {label: iris_df[iris_df["class"] == label] for label in class_labels}

    min_x, max_x, min_y, max_y, des_y1, des_y2 = decision_boundary(iris_df, classifier)
    des_x = [min_x, max_x]
    des_y = [des_y1, des_y2]

    col_1 = plot_settings[class_labels[0]]["color"]
    col_2 = plot_settings[class_labels[1]]["color"]
    plt.fill_between(des_x, min_y, des_y, color=col_1, alpha=0.25)
    plt.fill_between(des_x, des_y, max_y, color=col_2, alpha=0.25)

    plt.plot(des_x, des_y, color="navy", linewidth=2)

    if support_vectors:
        neg_sup_y = z_projection(min_x, max_x, classifier.w_, z_value=-1)
        pos_sup_y = z_projection(min_x, max_x, classifier.w_, z_value=1)
        plt.plot(des_x, neg_sup_y, color="blue", linewidth=1, linestyle="--")
        plt.plot(des_x, pos_sup_y, color="red", linewidth=1, linestyle="--")

    for label in class_labels:
        df = split_df[label]
        sett = plot_settings[label]
        plt.scatter(df["sepal length"], df["petal length"], **sett)

    plt.title("2 classes decision boundary")
    plt.xlabel("sepal length")
    plt.ylabel("petal length")
    plt.legend(loc="upper left")
    plt.show()


def decision_boundary(iris_df, classifier):
    b = 0.5

    min_x = iris_df["sepal length"].min() - b
    max_x = iris_df["sepal length"].max() + b

    min_y = iris_df["petal length"].min() - b
    max_y = iris_df["petal length"].max() + b

    w = classifier.w_
    des_y1, des_y2 = z_projection(min_x, max_x, w, z_value=0)

    return min_x, max_x, min_y, max_y, des_y1, des_y2


def z_projection(min_x, max_x, w, z_value=0):
    y1 = (z_value - w[0] - w[1] * min_x) / w[2]
    y2 = (z_value - w[0] - w[1] * max_x) / w[2]
    return y1, y2
