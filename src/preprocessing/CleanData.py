import numpy as np

def CheckData(df):

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nLabel Distribution:")
    print(df["Label"].value_counts())


def CleanData(df):

    # Drop columns that are not needed for Option A
    df.drop(
        columns=["patient_id,readmitted_30_days", "disease_risk_level"],
        inplace=True,
        errors="ignore"
    )

    # Rename target column
    df.rename(
        columns={"no_show": "Label"},
        inplace=True
    )

    # Replace infinity values with NaN
    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # Remove rows with missing values
    df.dropna(inplace=True)

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    print("\nDataset Cleaned Successfully..!")
    print("Dataset Shape:", df.shape)

    return df