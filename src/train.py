import os
import joblib
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Relative import using package structure
from src.preprocess import load_and_preprocess_data

# Initialize DAGsHub Tracking
dagshub.init(repo_owner='heshan.thilakawardena', repo_name='smartcare-noshow-ai', mlflow=True)

def train_and_evaluate():
    # Automatically create the 'models' output directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Path pointing to the CSV file inside the 'data' directory
    csv_path = 'data/smartcare_ai_dataset_1000.csv'
    
    X_train, X_test, y_train, y_test, preprocessor, _ = load_and_preprocess_data(csv_path)
    
    models = {
        'Random_Forest': RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42, eval_metric='logloss')
    }
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            pipeline.fit(X_train, y_train)
            
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            
            # Compute classification evaluation metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            
            print(f"\n--- {name} Results ---")
            print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
            
            # Log metrics and parameters to DAGsHub via MLflow
            mlflow.log_param("model_type", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", auc)
            
            # Save pipeline as .joblib model file
            model_filename = f"models/{name.lower()}_model.joblib"
            joblib.dump(pipeline, model_filename)
            mlflow.log_artifact(model_filename)
            print(f"Successfully saved trained pipeline to {model_filename}")

if __name__ == '__main__':
    train_and_evaluate()