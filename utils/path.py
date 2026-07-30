from pathlib import Path

BAS_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = BAS_DIR / "data" / "raw"/ "smartcare_ai_dataset_1000.csv"
MODEL_PATH = BAS_DIR / "models"
LOGI_REPORT_PATH = BAS_DIR / "reports" / "Logistic_Regression"
KNN_REPORT_PATH = BAS_DIR / "reports" / "KNN"
PROCESSED_DATA_PATH = BAS_DIR / "data" / "processed"