import dagshub
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Connect script to your friend's DagsHub repository
dagshub.init(
    repo_owner='heshanthilakawardena', 
    repo_name='smartcare-noshow-ai', 
    mlflow=True
)

# 2. Set experiment name
mlflow.set_experiment("D_tree_Model_Experiment")

# Load your dataset (replace with your actual data preprocessing/dataset)
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train & track experiment
with mlflow.start_run(run_name="Decision_Tree_Run_1"):
    max_depth = 5
    
    # Train model
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    acc = accuracy_score(y_test, clf.predict(X_test))
    
    # Log parameters & metrics to DagsHub Experiments tab
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", acc)
    
    # Log model artifact into DagsHub MLflow registry
    mlflow.sklearn.log_model(clf, "decision_tree_model")

    print(f"Model trained successfully! Accuracy: {acc:.4f}")
