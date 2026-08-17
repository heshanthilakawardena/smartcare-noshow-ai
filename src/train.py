import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from src.preprocess import PrepareData

# Initialize DAGsHub & MLflow Tracking
try:
    dagshub.init(repo_owner="heshan.thilakawardena", repo_name="smartcare-noshow-ai", mlflow=True)
except Exception as e:
    print(f"DAGsHub initialization warning: {e}")

# Resolve project paths dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "smartcare_ai_dataset_1000.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_and_evaluate():
    print(f"Loading raw dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    X_train, X_test, y_train, y_test, preprocessor, X_raw, y_raw = PrepareData(
        df=df,
        processed_data_path=PROCESSED_DIR,
        model_path=MODELS_DIR
    )

    #Define the Target Models
    models = {
        'Random_Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
    }

    # Target save paths matching expected names
    save_paths = {
        'Random_Forest': MODELS_DIR / "random_forest_model.joblib",
        'XGBoost': MODELS_DIR / "xgboost_model.joblib"
    }

    #Train & Log Models
    for name, clf in models.items():
        with mlflow.start_run(run_name=name):

            print(f"▶ Training Model: {name}")


            # Fit model on transformed preprocessed features
            clf.fit(X_train, y_train)

            # Predictions
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_proba)

            print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}\n")
            print(classification_report(y_test, y_pred, zero_division=0))

            # MLflow Logging
            mlflow.log_params(clf.get_params())
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", auc)

            # Save standalone model file
            joblib.dump(clf, save_paths[name])
            print(f"Successfully saved trained model to: {save_paths[name]}")

    print("\n Training complete!")


if __name__ == "__main__":
    train_and_evaluate()