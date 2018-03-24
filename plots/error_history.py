import matplotlib.pyplot as plt


def plot_error_history(errors_history):
    plt.plot(range(1, len(errors_history) + 1), errors_history, marker="o")
    plt.title("Errors history")
    plt.xlabel("Epochs")
    plt.ylabel("Number of misclassifications")
    plt.show()
