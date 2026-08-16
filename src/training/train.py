from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
import joblib


# =====================================
# Decision Tree Training
# =====================================

def TrainDecisionTree(X_train, y_train, model_path):

    model_path = Path(model_path)

    model_path.mkdir(
        parents=True,
        exist_ok=True
    )

    model_file = (
        model_path /
        "Smartcare_Decision_Tree_Model.joblib"
    )

    # Load existing model
    if model_file.exists():

        print("✅ Decision Tree model already exists.")
        print("Loading saved model...")

        model = joblib.load(
            model_file
        )

        return model

    # Train new model
    print("Training Decision Tree Model...")

    # Correctly initialize DecisionTreeClassifier
    model = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    train_accuracy = accuracy_score(
        y_train,
        model.predict(X_train)
    )

    print(
        f"Decision Tree Training Accuracy : {train_accuracy:.4f}"
    )

    # Save model
    joblib.dump(
        model,
        model_file
    )

    print(
        "✅ Decision Tree model saved successfully!"
    )

    return model