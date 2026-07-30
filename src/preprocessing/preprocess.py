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
    processed_data_exists,
)

from src.training.train import TrainModel
from src.training.evaluate import EvaluateModel

from utils.mlflow_tracker import MLflowTracker



def TrainingPipeline():


    print(
        "-----------------------------\n"
        "TRAINING PIPELINE ONLINE\n"
        "-----------------------------\n"
    )
    #Load Existing Processed Data
    if processed_data_exists(PROCESSED_DATA_PATH):

        print(
            "+ PROCESSED DATA FOUND\n"
            "+ LOADING PROCESSED DATA\n"
        )


        X, y = load_processed_data(
            PROCESSED_DATA_PATH
        )


        # PrepareData should handle
        # encoding + scaling
        (
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor,
            X,
            y

        ) = PrepareData(
            X.assign(Label=y),
            PROCESSED_DATA_PATH
        )

    else:


        print(
            "-----------------------------\n"
            "LOADING CSV FILE\n"
            "-----------------------------\n"
        )

        df = LoadData(RAW_DATA_PATH)

        print(
            "-----------------------------\n"
            "CLEANING DATA\n"
            "-----------------------------\n"
        )

        df = CleanData(df)



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



    # ==================================
    # Save Preprocessor
    # ==================================

    import joblib

    joblib.dump(

        preprocessor,

        "models/Smartcare_Preprocessor.joblib"

    )



    # ==================================
    # Train Model
    # ==================================

    print(
        "-----------------------------\n"
        "TRAINING LOGISTIC REGRESSION MODEL\n"
        "-----------------------------\n"
    )


    model = TrainModel(

        X_train,

        y_train,

        MODEL_PATH

    )



    # ==================================
    # Evaluation
    # ==================================

    print(
        "-----------------------------\n"
        "EVALUATING MODEL\n"
        "-----------------------------\n"
    )


    metrics = EvaluateModel(

        model,

        X_test,

        y_test,

        REPORT_PATH

    )



    # ==================================
    # MLflow Tracking
    # ==================================

    print(
        "-----------------------------\n"
        "TRACKING MODEL WITH MLFLOW\n"
        "-----------------------------\n"
    )


    MLflowTracker(

        model_name="Smartcare_Logistic_Regression_Model",

        metrics=metrics,

        REPORT_PATH=REPORT_PATH,

        MODEL_PATH=MODEL_PATH

    )



    print(
        "-----------------------------\n"
        "LOGISTIC REGRESSION TRAINING COMPLETED\n"
        "PIPELINE COMPLETED SUCCESSFULLY\n"
        "-----------------------------"
    )



if __name__ == "__main__":

    TrainingPipeline()