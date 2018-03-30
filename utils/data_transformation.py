def binarize_set_ver_label(row):
    binary_labels = {
        "Iris-setosa": -1,
        "Iris-versicolor": 1
    }
    return binary_labels[row["class"]]


def standardize_columns(df, columns):
    standardized_df = df.copy()
    for column in columns:
        x = standardized_df[column]
        standardized_df[column] = (x - x.mean()) / x.std()
    return standardized_df
