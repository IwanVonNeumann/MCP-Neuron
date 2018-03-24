import os
import pandas as pd

data_dir = "data"
data_file_name = "iris.data.csv"
data_file_path = os.path.join(data_dir, data_file_name)


def load_iris_data():
    columns = [
        "sepal length",
        "sepal width",
        "petal length",
        "petal width",
        "class"
    ]

    iris_df = pd.read_csv(data_file_path, header=None)
    iris_df.columns = columns

    return iris_df
