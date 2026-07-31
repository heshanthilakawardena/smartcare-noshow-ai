from utils.path import (
    RAW_DATA_PATH,
    MODEL_PATH,
    DECISION_REPORT_PATH,
    PROCESSED_DATA_PATH
)
from src.preprocessing.LoadData import LoadData
from src.preprocessing.CleanData import CleanData
from src.preprocessing.preprocess import PrepareData
from src.preprocessing.SaveObjects import (
    load_processed_data,
    processed_data_exists
)
from src.training.train import TrainDecisionTree
from src.training.evaluate import EvaluateDecisionTree
from utils.mlflow_tracker import MLflowTracker
from sklearn.model_selection import train_test_split
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

        # Recreate train/test splits from loaded processed data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Load the saved preprocessor
        preprocessor = joblib.load(
            Path(MODEL_PATH) / "Smartcare_Preprocessor.joblib"
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
            PROCESSED_DATA_PATH,
            MODEL_PATH
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

    decisiontree_model = TrainDecisionTree(
        X_train,
        y_train,
        MODEL_PATH
    )

    print("+ Evaluating Decision Tree model\n")

    decisiontree_metrics = EvaluateDecisionTree(
        decisiontree_model,
        X_test,
        y_test,
        DECISION_REPORT_PATH
    )

    print("+ MLFlow Tracking Decision Tree model\n")
    print(" ")

    # MLFLOW DECISION TREE
    MLflowTracker(
        model_name="Smartcare_Decision_Tree_Model",
        metrics=decisiontree_metrics,
        REPORT_PATH=DECISION_REPORT_PATH,
        MODEL_PATH=MODEL_PATH
    )

    print(
        "-----------------------------\n"
        "DECISION TREE MODEL TRAINED SUCCESSFULLY\n"
        "PIPELINE SHUT DOWN\n"
        "- ZAHRA ISMAIL -\n"
        "-----------------------------"
    )


if __name__ == "__main__":
    TrainingPipeline()