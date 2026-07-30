import os
import joblib
import pandas as pd
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score
)
# Load custom preprocessor from src/preprocess.py
from src.preprocess import load_and_preprocess_data

# 1. Connect script to DagsHub repository
dagshub.init(
    repo_owner='heshan.thilakawardena', 
    repo_name='smartcare-noshow-ai', 
    mlflow=True
)

# 2. Set experiment name
mlflow.set_experiment("smartcare-noshow-ai")

# 3. Load & Preprocess Healthcare Dataset (Replaces Iris Dataset)
data_path = os.path.join("data", "smartcare_ai_dataset_1000.csv")
X_train, X_test, y_train, y_test, preprocessor, X = load_and_preprocess_data(data_path)

# Apply ColumnTransformer preprocessing
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Ensure local 'models' directory exists
os.makedirs("models", exist_ok=True)

# 4. Hyperparameter Tuning & MLflow Experiment Tracking
depths_to_test = [3, 5, 7, 10, None]
best_f1 = 0
best_model = None

for depth in depths_to_test:
    run_name = f"Decision_Tree_depth_{depth if depth else 'unlimited'}"
    
    with mlflow.start_run(run_name=run_name):
        # Train model
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train_transformed, y_train)
        
        # Predictions & Probabilities
        preds = clf.predict(X_test_transformed)
        preds_proba = clf.predict_proba(X_test_transformed)
        
        # Calculate evaluation metrics
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        precision = precision_score(y_test, preds, average='weighted')
        recall = recall_score(y_test, preds, average='weighted')
        
        if preds_proba.shape[1] == 2:
            roc_auc = roc_auc_score(y_test, preds_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, preds_proba, multi_class='ovr')
        
        # Log parameters & metrics to DagsHub
        mlflow.log_param("model_type", "Decision_Tree")
        mlflow.log_param("max_depth", str(depth))
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Log model artifact to MLflow
        mlflow.sklearn.log_model(clf, f"decision_tree_depth_{depth}")
        
        print(f"Run '{run_name}' complete | Accuracy: {acc:.4f} | F1: {f1:.4f}")
        
        # Track best model based on F1-score
        if f1 > best_f1:
            best_f1 = f1
            best_model = clf

# 5. Save the best performing Decision Tree model to models/
if best_model is not None:
    model_path = os.path.join("models", "decision_tree_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"\nBest Decision Tree model saved to: {model_path} (F1 Score: {best_f1:.4f})")
