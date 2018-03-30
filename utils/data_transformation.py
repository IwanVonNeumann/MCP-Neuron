def binarize_class_labels(df):
    class_labels = df["class"].unique()
    if len(class_labels) != 2:
        raise ValueError("DataFrame of two classes expected")

    binary_labels = {
        class_labels[0]: -1,
        class_labels[1]: 1
    }
    df["binary class"] = df.apply(lambda row: binary_labels[row["class"]], axis=1)
    return df


def standardize_columns(df, columns):
    standardized_df = df.copy()
    for column in columns:
        x = standardized_df[column]
        standardized_df[column] = (x - x.mean()) / x.std()
    return standardized_df
