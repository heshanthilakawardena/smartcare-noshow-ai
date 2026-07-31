import joblib
from pathlib import Path
from utils.server_path import (
    LOGISTIC_MODEL_PATH,
    KNN_MODEL_PATH,
    PREPROCESSOR_PATH
)

# ==========================
# Load Models
# ==========================

print("Loading Models...")


models = {

    "Logistic Regression":
        joblib.load(LOGISTIC_MODEL_PATH),


    "KNN":
        joblib.load(KNN_MODEL_PATH)

}



print("✅ Logistic Regression Loaded")
print("Classes:", models["Logistic Regression"].classes_)
print("✅ KNN Loaded")
print("Classes:", models["KNN"].classes_)



# ==========================
# Load Preprocessor
# ==========================


preprocessor = joblib.load(
    PREPROCESSOR_PATH
)


print("✅ Preprocessor Loaded")