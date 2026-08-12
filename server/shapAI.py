import shap
import numpy as np
import re
import json


def clean_to_float_array(data):
    """Extracts float numbers from string brackets and converts to float64 NumPy array."""
    if data is None:
        return None

    if hasattr(data, "toarray"):
        data = data.toarray()

    arr = np.asarray(data)

    flat_list = []
    for item in arr.ravel():
        s = str(item)
        match = re.search(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", s)
        if match:
            flat_list.append(float(match.group(0)))
        else:
            flat_list.append(0.0)

    result = np.array(flat_list, dtype=np.float64)
    if arr.ndim == 1:
        return result.reshape(1, -1)
    return result.reshape(arr.shape)


def repair_xgboost_base_score(model):
    """
    Fixes the string bracket issue '[4.950002E-1]' in XGBoost's C++ booster configuration
    so SHAP's XGBTreeModelLoader can safely convert base_score to float.
    """
    try:
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        
        # 1. Update booster attribute directly
        booster.set_attr(base_score="0.4950002")

        # 2. Re-parse and clean internal JSON configuration
        cfg = json.loads(booster.save_config())
        if "learner" in cfg and "learner_model_param" in cfg["learner"]:
            raw_score = str(cfg["learner"]["learner_model_param"].get("base_score", ""))
            clean_score = re.sub(r"[\[\]]", "", raw_score)
            cfg["learner"]["learner_model_param"]["base_score"] = clean_score
            booster.load_config(json.dumps(cfg))
    except Exception as e:
        print(f"⚠️ Booster repair notice: {e}")


def explain_prediction(
        model,
        model_name,
        processed_input,
        feature_names,
        background_data
):
    processed_input = clean_to_float_array(processed_input)
    if background_data is not None:
        background_data = clean_to_float_array(background_data)


    # Logistic Regression SHAP

    if model_name == "Logistic Regression":
        explainer = shap.LinearExplainer(
            model,
            background_data
        )
        shap_values = explainer(processed_input)
        values = shap_values.values[0]


    # Random Forest & Decision Tree SHAP

    elif model_name in ["Random Forest", "Decision Tree"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(processed_input)

        try:
            if hasattr(shap_values, "values"):
                vals = shap_values.values
                if len(vals.shape) == 3:
                    values = vals[0, :, 1]
                elif len(vals.shape) == 2:
                    values = vals[0]
                else:
                    values = vals
            else:
                values = shap_values[0]
        except Exception:
            values = np.asarray(shap_values)[0]


    # XGBoost SHAP

    elif model_name == "XGBoost":
        # Apply repair before passing to SHAP
        repair_xgboost_base_score(model)

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(processed_input)
            
            if hasattr(shap_values, "values"):
                vals = shap_values.values
                if len(vals.shape) == 3:
                    values = vals[0, :, 1]
                elif len(vals.shape) == 2:
                    values = vals[0]
                else:
                    values = vals
            else:
                values = shap_values[0]
        except Exception as err:
            print(f" TreeExplainer failed ({err}), switching to KernelExplainer fallback...")
            # Fallback that bypasses XGBoost C++ model loaders entirely
            explainer = shap.KernelExplainer(
                model.predict_proba,
                background_data
            )
            shap_values = explainer.shap_values(processed_input)
            if isinstance(shap_values, list):
                values = shap_values[1][0]
            else:
                values = shap_values[0]


    # KNN SHAP

    else:
        explainer = shap.KernelExplainer(
            model.predict_proba,
            background_data
        )
        try:
            shap_values = explainer(processed_input)
            if hasattr(shap_values, "values") and len(shap_values.values.shape) == 3:
                values = shap_values.values[0, :, 1]
            else:
                values = shap_values[0]
        except Exception:
            shap_values = explainer.shap_values(processed_input)
            if isinstance(shap_values, list):
                values = shap_values[1][0]
            else:
                values = shap_values[0]


    # Create Explanation Dict

    explanation = {}
    values = np.asarray(values).ravel()

    for name, value in zip(feature_names, values):
        clean_name = (
            str(name)
            .replace("onehot__", "")
            .replace("numeric__", "")
            .replace("_", " ")
            .title()
        )
        explanation[clean_name] = float(value)

    # Top 5 Features
    explanation = dict(
        sorted(
            explanation.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
    )

    return explanation