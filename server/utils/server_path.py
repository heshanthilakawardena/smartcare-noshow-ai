from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]


MODEL_PATH = BASE_DIR / "model"


LOGISTIC_MODEL_PATH = (
    MODEL_PATH /
    "Smartcare_Logistic_Regression_Model.joblib"
)


KNN_MODEL_PATH = (
    MODEL_PATH /
    "Smartcare_KNN_Model.joblib"
)


PREPROCESSOR_PATH = (
    MODEL_PATH /
    "Smartcare_Preprocessor.joblib"
)

STEAMLIT = BASE_DIR / "ui" / "dashboard.py"