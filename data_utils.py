def binarize_set_ver_label(row):
    binary_labels = {
        "Iris-setosa": -1,
        "Iris-versicolor": 1
    }
    return binary_labels[row["class"]]


def standardize(df):
    return (df - df.mean()) / df.std()
