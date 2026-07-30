from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Common Evaluation Function

def EvaluateModel(model, model_name, X_test, y_test, report_path):

    report_path = Path(report_path)

    report_path.mkdir(
        parents=True,
        exist_ok=True
    )


    print(
        f"\n========== Evaluating {model_name} ==========\n"
    )

    # Predictions
    y_pred = model.predict(
        X_test
    )


    # Probability for ROC
    y_prob = model.predict_proba(
        X_test
    )[:,1]

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )


    f1 = f1_score(
        y_test,
        y_pred
    )


    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )



    metrics = {

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")



    # -----------------------------
    # Save Metrics
    # -----------------------------

    with open(report_path / "model_metrics.txt","w") as f:

        f.write(f"========== {model_name} Evaluation ==========\n\n")
        for key,value in metrics.items():
            f.write(
                f"{key}: {value:.4f}\n"
            )

    # Classification Report

    report = classification_report(

        y_test,

        y_pred,

        target_names=[
            "Attended",
            "No Show"
        ]

    )

    print("\nClassification Report\n",report)

    with open(report_path / "classification_report.txt","w") as f:
        f.write(report)

    # Confusion Matrix

    cm = confusion_matrix(

        y_test,

        y_pred

    )



    plt.figure(

        figsize=(6,5)

    )


    sns.heatmap(

        cm,
        annot=True,
        fmt="d",
        xticklabels=[
            "Attended",
            "No Show"
        ],
        yticklabels=[
            "Attended",
            "No Show"
        ]
    )


    plt.xlabel(
        "Predicted"
    )


    plt.ylabel(
        "Actual"
    )


    plt.title(
        f"{model_name} - Confusion Matrix"
    )


    plt.tight_layout()


    plt.savefig(
        report_path / "confusion_matrix.png",
        dpi=300

    )


    plt.close()



    print("✅ Confusion Matrix saved")



    # -----------------------------
    # ROC Curve
    # -----------------------------

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    plt.figure(
        figsize=(7,5)
    )

    plt.plot(
        fpr,
        tpr,
        label=f"AUC={roc_auc:.3f}"
    )

    plt.plot(
        [0,1],
        [0,1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        f"{model_name} - ROC Curve"
    )

    plt.legend(

        loc="lower right"

    )

    plt.tight_layout()

    plt.savefig(
        report_path / "roc_curve.png",
        dpi=300
    )


    plt.close()



    print("✅ ROC Curve saved")


    return metrics

#Logistic Regression Evaluation

def EvaluateLogisticRegression(
        model,
        X_test,
        y_test,
        report_path
):

    return EvaluateModel(

        model,
        "Logistic Regression",
        X_test,
        y_test,
        report_path

    )


#KNN Evaluation
def EvaluateKNN(
        model,
        X_test,
        y_test,
        report_path
):

    return EvaluateModel(
        model,
        "KNN",
        X_test,
        y_test,
        report_path
    )