from sklearn.preprocessing import StandardScaler


def binarize_class_labels(df):
    class_labels = df["class"].unique()
    if len(class_labels) != 2:
        raise ValueError("DataFrame of two classes expected")

    negative, positive = class_labels

    encoded_labels = {
        negative: -1,
        positive: 1
    }

    df["binary class"] = df.apply(lambda row: encoded_labels[row["class"]], axis=1)

    return df


def standardize(df, columns):
    standardized_df = df.copy()
    scaler = StandardScaler()
    for column in columns:
        standardized_df[column] = scaler.fit_transform(standardized_df[[column]])
    return standardized_df
