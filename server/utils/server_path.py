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

XGBOOST_MODEL_PATH = (
    MODEL_PATH /
    "xgboost_model.joblib"
)

RANDOM_FOREST_MODEL_PATH = (
    MODEL_PATH / 
    "random_forest_model.joblib"
)

DISICION_TREE_MODEL_PATH = (
    MODEL_PATH / 
    "Smartcare_Decision_Tree_Model.joblib"
)
PREPROCESSOR_PATH = (
    MODEL_PATH /
    "Smartcare_Preprocessor.joblib"
)

SHAP_BACKGROUND_PATH = (
    MODEL_PATH /
    "Smartcare_SHAP_Background.joblib"
)

STEAMLIT = BASE_DIR / "ui" / "dashboard.py"

# UI Paths

TEMPLATE_PATH = (
    BASE_DIR /
    "ui" /
    "template"
)


STATIC_PATH = (
    BASE_DIR /
    "ui" /
    "static"
)