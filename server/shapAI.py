import shap


def explain_prediction(
        model,
        model_name,
        processed_input,
        feature_names,
        background_data
):


    # ==========================
    # Logistic Regression SHAP
    # ==========================

    if model_name == "Logistic Regression":


        explainer = shap.LinearExplainer(
            model,
            background_data
        )


        shap_values = explainer(
            processed_input
        )


        values = shap_values.values[0]

    # ==========================
    # XGBoost & Random Forest SHAP
    # ==========================

    elif model_name in ["XGBoost", "Random Forest"]:

        explainer = shap.TreeExplainer(
            model
        )

        shap_values = explainer(
            processed_input
        )

        try:

            # Binary classification
            values = shap_values.values[0, :, 1]

        except Exception:

            values = shap_values.values[0]



    # ==========================
    # KNN SHAP
    # ==========================

    else:


        explainer = shap.KernelExplainer(
            model.predict_proba,
            background_data
        )


        try:

            # New SHAP versions

            shap_values = explainer(
                processed_input
            )


            values = shap_values.values[0, :, 1]



        except Exception:


            # Older SHAP versions

            shap_values = explainer.shap_values(
                processed_input
            )


            if isinstance(shap_values, list):

                values = shap_values[1][0]


            else:

                values = shap_values[0]



    # ==========================
    # Create Explanation
    # ==========================

    explanation = {}



    for name, value in zip(
        feature_names,
        values
    ):


        # Clean feature names

        clean_name = (
            name
            .replace("onehot__", "")
            .replace("numeric__", "")
            .replace("_", " ")
            .title()
        )


        explanation[clean_name] = float(value)



    # ==========================
    # Select Top 5 Important Features
    # ==========================

    explanation = dict(

        sorted(
            explanation.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

    )



    return explanation