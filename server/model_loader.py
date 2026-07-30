import joblib
from pathlib import Path
from utils.path import MODEL_PATH


# ==========================
# Model Paths
# ==========================

LOGISTIC_PATH = (
    MODEL_PATH /
    "Smartcare_Logistic_Regression_Model.joblib"
)


KNN_PATH = (
    MODEL_PATH /
    "Smartcare_KNN_Model.joblib"
)


PREPROCESSOR_PATH = (
    MODEL_PATH /
    "Smartcare_Preprocessor.joblib"
)



# ==========================
# Load Models
# ==========================

print("Loading Models...")


models = {

    "Logistic Regression":
        joblib.load(LOGISTIC_PATH),


    "KNN":
        joblib.load(KNN_PATH)

}



print("✅ Logistic Regression Loaded")
print("✅ KNN Loaded")



# ==========================
# Load Preprocessor
# ==========================


preprocessor = joblib.load(
    PREPROCESSOR_PATH
)


print("✅ Preprocessor Loaded")