import os
import pandas as pd

data_dir = "data"
data_file_name = "iris.data.csv"
data_file_path = os.path.join(data_dir, data_file_name)


def load_iris_data():
    return pd.read_csv(data_file_path)
