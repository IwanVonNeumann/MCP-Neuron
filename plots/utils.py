iris_plot_settings = {
    "Iris-setosa": {
        "color": "blue",
        "marker": "x",
        "label": "setosa"
    },
    "Iris-versicolor": {
        "color": "red",
        "marker": "o",
        "label": "versicolor"
    },
    "Iris-virginica": {
        "color": "lime",
        "marker": "^",
        "label": "virginica"
    }
}


def get_decision_boundary_color(labels):
    if set(labels) == {"Iris-setosa", "Iris-versicolor"}:
        return "navy"
    if set(labels) == {"Iris-setosa", "Iris-virginica"}:
        return "navy"
    if set(labels) == {"Iris-versicolor", "Iris-virginica"}:
        return "maroon"
