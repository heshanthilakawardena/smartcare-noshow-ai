import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.preprocess import load_and_preprocess_data

# Initialize DAGsHub & MLflow Tracking
try:
    dagshub.init(repo_owner="heshan.thilakawardena", repo_name="smartcare-noshow-ai", mlflow=True)
except Exception as e:
    print(f"DAGsHub initialization warning: {e}")

DATA_PATH = "data/smartcare_ai_dataset_1000.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_evaluate():
    # 1. Load Data & Preprocessor
    X_train, X_test, y_train, y_test, preprocessor, _ = load_and_preprocess_data(DATA_PATH)

    # 2. Define Models with Tuned Hyperparameters
    models = {
        'Random_Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=6,              # Constrained depth to prevent overfitting
            min_samples_split=5,
            class_weight='balanced',  # Balance class weights
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=3,              # Shallow depth ideal for 1000 tabular rows
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        ),
        'Decision_Tree_Zahra': DecisionTreeClassifier(
            max_depth=5,              # Controlled depth to prevent over-splitting
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42
        )
    }

    # Model save mapping
    save_paths = {
        'Random_Forest': os.path.join(MODELS_DIR, "random_forest_model.joblib"),
        'XGBoost': os.path.join(MODELS_DIR, "xgboost_model.joblib"),
        'Decision_Tree_Zahra': os.path.join(MODELS_DIR, "decision_tree_zahra_model.joblib")
    }

    # 3. Train & Evaluate Models
    for name, clf in models.items():
        with mlflow.start_run(run_name=name):
            # Create full sklearn Pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', clf)
            ])

            # Fit pipeline
            pipeline.fit(X_train, y_train)

            # Predictions
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred

            # Evaluation Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_proba)

            # Print Results to Terminal
            print(f"\n--- {name} Results ---")
            print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

            # MLflow Logging
            mlflow.log_params(clf.get_params())
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", auc)

            # Save Trained Pipeline Locally
            joblib.dump(pipeline, save_paths[name])
            print(f"Successfully saved trained pipeline to {save_paths[name]}")

if __name__ == "__main__":
    train_and_evaluate()