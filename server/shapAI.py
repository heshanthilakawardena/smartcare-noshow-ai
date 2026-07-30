import shap
import numpy as np


def explain_prediction(
        model,
        model_name,
        processed_input,
        feature_names
):


    if model_name == "Logistic Regression":

        explainer = shap.LinearExplainer(
            model,
            processed_input
        )


        shap_values = explainer(
            processed_input
        )


        values = shap_values.values[0]



    else:

        explainer = shap.KernelExplainer(
            model.predict_proba,
            processed_input
        )

        try:
            # New SHAP versions
            shap_values = explainer(processed_input)
            values = shap_values.values[0, :, 1]

        except Exception:
            # Older SHAP versions
            shap_values = explainer.shap_values(processed_input)

            if isinstance(shap_values, list):
                values = shap_values[1][0]
            else:
                values = shap_values[0]



    explanation = {}


    for name,value in zip(
        feature_names,
        values
    ):

        explanation[name] = float(value)



    # sort important features

    explanation = dict(

        sorted(
            explanation.items(),
            key=lambda x:abs(x[1]),
            reverse=True
        )[:5]

    )


    return explanation