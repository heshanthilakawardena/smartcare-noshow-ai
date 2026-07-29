import os
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score
)

# 1. Connect script to DagsHub repository
dagshub.init(
    repo_owner='heshan.thilakawardena', 
    repo_name='smartcare-noshow-ai', 
    mlflow=True
)

# 2. Set experiment name (matches your team's experiment board)
mlflow.set_experiment("smartcare-noshow-ai")

# --- TODO: Replace Iris with your actual preprocessed healthcare dataset ---
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# --------------------------------------------------------------------------

# 3. Train & track experiment
with mlflow.start_run(run_name="Decision_Tree_Zahra"):
    max_depth = 5
    
    # Train model
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    
    # Generate predictions & probabilities
    preds = clf.predict(X_test)
    preds_proba = clf.predict_proba(X_test)
    
    # Calculate evaluation metrics
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    precision = precision_score(y_test, preds, average='weighted')
    recall = recall_score(y_test, preds, average='weighted')
    roc_auc = roc_auc_score(y_test, preds_proba, multi_class='ovr')
    
    # Log parameters (matching your team's table columns)
    mlflow.log_param("model_type", "Decision_Tree")
    mlflow.log_param("max_depth", max_depth)
    
    # Log all metrics to DagsHub
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", roc_auc)
    
    # Log model artifact into DagsHub MLflow registry
    mlflow.sklearn.log_model(clf, "decision_tree_model")

    print(f"Model trained and logged successfully! Accuracy: {acc:.4f}")