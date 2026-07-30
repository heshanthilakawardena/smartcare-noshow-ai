from utils.path import (
    RAW_DATA_PATH,
    MODEL_PATH,
    REPORT_PATH,
    PROCESSED_DATA_PATH
)
from src.preprocessing.LoadData import LoadData
from src.preprocessing.CleanData import CleanData
from src.preprocessing.preprocess import PrepareData
from src.preprocessing.SaveObjects import (
    load_processed_data,
    processed_data_exists
)
from src.training.train import (
    TrainKNN,
    TrainLogisticRegression
)
from src.training.evaluate import (
    EvaluateLogisticRegression,
    EvaluateKNN
)
from utils.mlflow_tracker import MLflowTracker
import joblib
from pathlib import Path



def TrainingPipeline():


    print(
        "-----------------------------\n"
        "TRAINING PIPELINE ONLINE\n"
        "-----------------------------\n"
    )

    if processed_data_exists(PROCESSED_DATA_PATH):

        print(
            "+ PROCESSED DATA FOUND\n"
            "+ LOADING DATA\n"
        )


        X, y = load_processed_data(
            PROCESSED_DATA_PATH
        )

    else:


        print(
            "-----------------------------\n"
            "LOADING CSV FILE\n"
            "-----------------------------\n"
        )

        df = LoadData(
            RAW_DATA_PATH
        )

        print(
            "-----------------------------\n"
            "CLEANING DATA\n"
            "-----------------------------\n"
        )


        df = CleanData(
            df
        )



        print(
            "-----------------------------\n"
            "PREPARING DATA\n"
            "-----------------------------\n"
        )


        (
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor,
            X,
            y

        ) = PrepareData(

            df,

            PROCESSED_DATA_PATH

        )

    Path(MODEL_PATH).mkdir(
        parents=True,
        exist_ok=True
    )


    joblib.dump(
        preprocessor,
        Path(MODEL_PATH) /
        "Smartcare_Preprocessor.joblib"
    )


    print("✅ Preprocessor saved\n")

    print(
        "-----------------------------\n"
        "TRAINING EVALUATING & MLFLOW Tracking MODELS\n"
        "-----------------------------\n"
    )

    logistic_model = TrainLogisticRegression(
        X_train,
        y_train,
        MODEL_PATH
    )

    print("+ Evaluating logistic regression model\n")

    logistic_metrics = EvaluateLogisticRegression(
        logistic_model,
        X_test,
        y_test,
        REPORT_PATH
    )

    print("+ MLFlow Tracking logistic regression model\n")
    print(" ")

    # MLFLOW LOGISTIC
    MLflowTracker(
        model_name="Smartcare_Logistic_Regression_Model",
        metrics=logistic_metrics,
        REPORT_PATH=Path(REPORT_PATH) / "Logistic_Regression",
        MODEL_PATH=MODEL_PATH
    )

    # TRAIN KNN
    print("+ KNN model training\n")

    knn_model = TrainKNN(
        X_train,
        y_train,
        MODEL_PATH
    )

    # EVALUATE KNN
    print("+ Evaluating KNN model\n")

    knn_metrics = EvaluateKNN(
        knn_model,
        X_test,
        y_test,
        REPORT_PATH
    )

    print("+ MLFlow Tracking KNN model\n")
    print(" ")

    MLflowTracker(
        model_name="Smartcare_KNN_Model",
        metrics=knn_metrics,
        REPORT_PATH=Path(REPORT_PATH) / "KNN",
        MODEL_PATH=MODEL_PATH
    )

    print(
        "-----------------------------\n"
        "BOTH MODELS TRAINED SUCCESSFULLY\n"
        "PIPELINE SHUT DOWN\n"
        "- BINARA WIJEWICKRAMA -\n"
        "-----------------------------"
    )


if __name__ == "__main__":

    TrainingPipeline()