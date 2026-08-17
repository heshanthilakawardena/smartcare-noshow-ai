
import pandas as pd
from pathlib import Path

def processed_data_exists(path):

    file_path = (
        path /
        "smartcare_processed_dataset.csv"
    )
    
    return file_path.exists()


def load_processed_data(path):

    file_path = (
        path /
        "smartcare_processed_dataset.csv"
    )


    df = pd.read_csv(
        file_path
    )


    X = df.drop(
        "Label",
        axis=1
    )


    y = df["Label"]


    print("+ Processed data loaded..!!")


    return X, y