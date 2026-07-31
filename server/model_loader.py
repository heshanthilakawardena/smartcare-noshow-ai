import joblib
from pathlib import Path
import xgboost
from utils.server_path import (
    LOGISTIC_MODEL_PATH,
    KNN_MODEL_PATH,
    XGBOOST_MODEL_PATH,
    RANDOM_FOREST_MODEL_PATH,
    PREPROCESSOR_PATH,
    SHAP_BACKGROUND_PATH
)

# ==========================
# Load Models
# ==========================

print("Loading Models...")


models = {

    "Logistic Regression":
        joblib.load(LOGISTIC_MODEL_PATH),


    "KNN":
        joblib.load(KNN_MODEL_PATH),

    "XGBoost":
        joblib.load(XGBOOST_MODEL_PATH),

    "Random Forest":
            joblib.load(RANDOM_FOREST_MODEL_PATH)

}



print("✅ Logistic Regression Loaded")
print("Classes:", models["Logistic Regression"].classes_)

print("✅ KNN Loaded")
print("Classes:", models["KNN"].classes_)

print("✅ XGBoost Loaded")
print("Classes:", models["XGBoost"].classes_)

print("✅ Random Forest Loaded")
print("Classes:", models["Random Forest"].classes_)


# ==========================
# Load Preprocessor
# ==========================


preprocessor = joblib.load(
    PREPROCESSOR_PATH
)


print("✅ Preprocessor Loaded")

background_data = joblib.load(
    SHAP_BACKGROUND_PATH
)