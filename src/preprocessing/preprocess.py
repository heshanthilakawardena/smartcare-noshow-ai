from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder
)

from sklearn.compose import ColumnTransformer

from pathlib import Path

import pandas as pd
import joblib



def PrepareData(df, processed_data_path, model_path):


    print("\n+ Starting Data Preparation...\n")


    # -----------------------------
    # Clean Column Names
    # -----------------------------

    df.columns = (
        df.columns
        .str.strip()
    )



    # -----------------------------
    # Remove Unnecessary Columns
    # -----------------------------

    remove_columns = [

        "readmitted_30_days",

        "disease_risk_level"

    ]


    df.drop(

        columns=[
            col for col in remove_columns
            if col in df.columns
        ],

        inplace=True

    )



    # -----------------------------
    # Appointment Date Feature Engineering
    # -----------------------------


    df["appointment_date"] = pd.to_datetime(

        df["appointment_date"]

    )


    df["appointment_year"] = (

        df["appointment_date"]
        .dt.year

    )


    df["appointment_month"] = (

        df["appointment_date"]
        .dt.month

    )


    df["appointment_day"] = (

        df["appointment_date"]
        .dt.day

    )


    df["appointment_dayofweek"] = (

        df["appointment_date"]
        .dt.dayofweek

    )


    df["appointment_weekend"] = (

        df["appointment_dayofweek"] >= 5

    ).astype(int)



    # Remove original date

    df.drop(

        columns=["appointment_date"],

        inplace=True

    )



    # -----------------------------
    # Save Processed Dataset
    # -----------------------------

    save_processed_data(

        df,

        processed_data_path

    )



    # -----------------------------
    # Split Features and Label
    # -----------------------------


    X = df.drop(

        "Label",

        axis=1

    )


    y = df["Label"]



    print(
        "Classes:",
        y.value_counts()
    )



    # -----------------------------
    # Encoding Columns
    # -----------------------------


    onehot_columns = [

        "gender",

        "blood_group",

        "department",

        "diagnosis",

        "room_type",

        "payment_method"

    ]



    ordinal_columns = [

        "appointment_status",

        "payment_status"

    ]



    numeric_columns = [

        col for col in X.columns

        if col not in
        onehot_columns + ordinal_columns

    ]



    # -----------------------------
    # Category Order
    # -----------------------------


    appointment_order = [

        "Scheduled",

        "Completed",

        "Cancelled",

        "No-Show"

    ]



    payment_order = [

        "Paid",

        "Partially Paid",

        "Unpaid"

    ]



    # -----------------------------
    # Preprocessor
    # -----------------------------


    preprocessor = ColumnTransformer(

        transformers=[


            (

                "onehot",

                OneHotEncoder(

                    handle_unknown="ignore"

                ),

                onehot_columns

            ),



            (

                "ordinal",

                OrdinalEncoder(

                    categories=[

                        appointment_order,

                        payment_order

                    ]

                ),

                ordinal_columns

            ),



            (

                "numeric",

                StandardScaler(),

                numeric_columns

            )

        ]

    )



    # -----------------------------
    # Train Test Split
    # -----------------------------


    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42,

        stratify=y

    )



    # -----------------------------
    # Fit preprocessing ONLY TRAIN
    # -----------------------------


    X_train = preprocessor.fit_transform(

        X_train

    )


    X_test = preprocessor.transform(

        X_test

    )



    # -----------------------------
    # Save Preprocessor
    # -----------------------------


    model_path = Path(model_path)


    model_path.mkdir(

        parents=True,

        exist_ok=True

    )


    joblib.dump(

        preprocessor,

        model_path /

        "Smartcare_Preprocessor.joblib"

    )



    print(
        "\n+ Data Preparation Completed"
    )



    return (

        X_train,

        X_test,

        y_train,

        y_test,

        preprocessor,

        X,

        y

    )





def save_processed_data(df, path):


    path = Path(path)


    path.mkdir(

        parents=True,

        exist_ok=True

    )



    df.to_csv(

        path /

        "smartcare_processed_dataset.csv",

        index=False

    )


    print(
        "+ Processed dataset saved"
    )