import joblib
import json
from pathlib import Path
import numpy as np
import re
import xgboost
from sklearn.compose import ColumnTransformer
from utils.server_path import (
    LOGISTIC_MODEL_PATH,
    KNN_MODEL_PATH,
    XGBOOST_MODEL_PATH,
    RANDOM_FOREST_MODEL_PATH,
    DISICION_TREE_MODEL_PATH,
    PREPROCESSOR_PATH,
    SHAP_BACKGROUND_PATH
)


# Compatibility & XGBoost Patch Helpers

def patch_xgboost_base_score(model):
    """Fixes internal bracketed string base_score in XGBoost models for SHAP compatibility."""
    try:
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        config = json.loads(booster.save_config())
        
        # Access learner model parameters where base_score is stored
        param_section = config.get("learner", {}).get("learner_model_param", {})
        if "base_score" in param_section:
            raw_val = str(param_section["base_score"])
            match = re.search(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", raw_val)
            if match:
                clean_val = match.group(0)
                config["learner"]["learner_model_param"]["base_score"] = clean_val
                booster.load_config(json.dumps(config))
                print(f"Patched XGBoost base_score: {raw_val} -> {clean_val}")
    except Exception as e:
        print(f"Could not patch XGBoost base_score automatically: {e}")


def patch_column_transformer(obj):
    """Recursively patches missing attributes in ColumnTransformers across scikit-learn versions."""
    if isinstance(obj, ColumnTransformer):
        if not hasattr(obj, '_name_to_fitted_passthrough'):
            setattr(obj, '_name_to_fitted_passthrough', {})
            
    if hasattr(obj, 'named_steps'):
        for step in obj.named_steps.values():
            patch_column_transformer(step)
    elif hasattr(obj, 'transformers_'):
        for _, trans, _ in obj.transformers_:
            patch_column_transformer(trans)


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
        # Match scientific notation or standard floats inside any bracket/quote wrappers
        match = re.search(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", s)
        if match:
            flat_list.append(float(match.group(0)))
        else:
            flat_list.append(0.0)

    result = np.array(flat_list, dtype=np.float64)
    if arr.ndim > 1:
        return result.reshape(arr.shape)
    elif len(arr.shape) == 1:
        return result.reshape(1, -1)
    return result



#Load Models


print("Loading Models...")

logistic_model = joblib.load(LOGISTIC_MODEL_PATH)
if not hasattr(logistic_model, 'multi_class'):
    setattr(logistic_model, 'multi_class', 'auto')

# Load and patch XGBoost model
xgb_model = joblib.load(XGBOOST_MODEL_PATH)
patch_xgboost_base_score(xgb_model)

models = {
    "Logistic Regression": logistic_model,
    "KNN": joblib.load(KNN_MODEL_PATH),
    "XGBoost": xgb_model,
    "Random Forest": joblib.load(RANDOM_FOREST_MODEL_PATH),
    "Decision Tree": joblib.load(DISICION_TREE_MODEL_PATH)
}

print("Logistic Regression Loaded")
print("KNN Loaded")
print("XGBoost Loaded")
print("Random Forest Loaded")
print("Decision Tree Loaded")



# Load Preprocessor & Background Data


preprocessor = joblib.load(PREPROCESSOR_PATH)
patch_column_transformer(preprocessor)
print("Preprocessor Loaded")

raw_background = joblib.load(SHAP_BACKGROUND_PATH)
background_data = clean_to_float_array(raw_background)
print("SHAP Background Data Cleaned & Loaded")