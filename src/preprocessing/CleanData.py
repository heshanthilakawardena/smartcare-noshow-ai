import numpy as np

def CheckData(df):

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nLabel Distribution:")
    print(df["Label"].value_counts())


def CleanData(df):

    # Drop columns that are not needed
    df.drop(
        columns = [
        "patient_id",
        "record_id",
        "appointment_status",
        "admitted",
        "blood_group",
        "room_type",
        "length_of_stay_days",
        "previous_admissions",
        "lab_tests_count",
        "treatments_count",
        "room_charge_lkr",
        "lab_charge_lkr",
        "medicine_charge_lkr",
        "total_bill_lkr",
        "payment_status",
        "payment_method",
        "readmitted_30_days",
        "disease_risk_level",
        "systolic_bp",
        "diastolic_bp",
        "blood_sugar_mg_dl",
        "cholesterol_mg_dl",
    ],
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