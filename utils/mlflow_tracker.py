import os
from datetime import datetime
from pathlib import Path
import mlflow
import dagshub
from dotenv import load_dotenv

load_dotenv()

def SetupMLflow():

    username = os.getenv("D_USERNAME")
    repo_name = os.getenv("D_REPO")


    if not username or not repo_name:

        print("+ MLflow disabled: DagsHub credentials missing")
        return False

    dagshub.init(
        repo_owner=username,
        repo_name=repo_name,
        mlflow=False
    )

    mlflow.set_tracking_uri(f"https://dagshub.com/{username}/{repo_name}.mlflow")

    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "SmartCare_No_Show_Prediction"
    )

    mlflow.set_experiment(

        experiment_name

    )

    print(f"+ MLflow Ready : {experiment_name}")

    return True


def MLflowTracker(
        model_name,
        metrics,
        REPORT_PATH,
        MODEL_PATH
):


    if not SetupMLflow():

        return

    MODEL_PATH = Path(MODEL_PATH)
    REPORT_PATH = Path(REPORT_PATH)

    model_file = (MODEL_PATH / f"{model_name}.joblib")

    run_name = (
        f"{model_name}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    with mlflow.start_run(
        run_name=run_name
    ):

        # Parameters
        mlflow.log_param(

            "model_name",
            model_name
        )

        if "Logistic" in model_name:

            mlflow.log_param(
                "algorithm",
                "Logistic Regression"
            )

        elif "KNN" in model_name:

            mlflow.log_param(
                "algorithm",
                "K-Nearest Neighbors"
            )

        mlflow.log_param(
            "problem_type",
            "Binary Classification"
        )

        mlflow.log_param(
            "dataset",
            "SmartCare Appointment No Show"
        )

        # Metrics
        for key,value in metrics.items():

            mlflow.log_metric(
                key,
                float(value)
            )

        # Model Artifact
        if model_file.exists():

            mlflow.log_artifact(
                str(model_file),
                artifact_path="models"
            )

            print(
                f"+ {model_name} uploaded"
            )

        else:

            print("+ Model file missing:",model_file)

        # Evaluation Artifacts
        if REPORT_PATH.exists():

            mlflow.log_artifacts(
                str(REPORT_PATH),
                artifact_path="evaluation"
            )

            print(f"+ {model_name} reports uploaded")

    print(f"✅ MLflow completed for {model_name}")