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


        values = explainer.shap_values(

            processed_input

        )[1][0]



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