from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
import joblib



# =====================================
# Logistic Regression Training
# =====================================

def TrainLogisticRegression(X_train, y_train, model_path):

    model_path = Path(model_path)

    model_path.mkdir(
        parents=True,
        exist_ok=True
    )


    model_file = (
        model_path /
        "Smartcare_Logistic_Regression_Model.joblib"
    )


    # Load existing model
    if model_file.exists():

        print("✅ Logistic Regression model already exists.")
        print("Loading saved model...")

        model = joblib.load(
            model_file
        )

        return model



    # Train new model

    print("Training Logistic Regression Model...")


    model = LogisticRegression(

        max_iter=1000,

        class_weight="balanced",

        solver="saga",

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
        f"Logistic Regression Training Accuracy : {train_accuracy:.4f}"
    )


    # Save model

    joblib.dump(

        model,

        model_file

    )


    print(
        "✅ Logistic Regression model saved successfully!"
    )


    return model





# =====================================
# KNN Training
# =====================================

def TrainKNN(X_train, y_train, model_path):

    model_path = Path(model_path)

    model_path.mkdir(
        parents=True,
        exist_ok=True
    )


    model_file = (
        model_path /
        "Smartcare_KNN_Model.joblib"
    )


    # Load existing model

    if model_file.exists():

        print("✅ KNN model already exists.")
        print("Loading saved model...")

        model = joblib.load(
            model_file
        )

        return model



    # Train new model

    print("Training KNN Model...")


    model = KNeighborsClassifier(

        n_neighbors=5,

        metric="minkowski",

        weights="distance"

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
        f"KNN Training Accuracy : {train_accuracy:.4f}"
    )



    # Save model

    joblib.dump(

        model,

        model_file

    )


    print(
        "✅ KNN model saved successfully!"
    )


    return model