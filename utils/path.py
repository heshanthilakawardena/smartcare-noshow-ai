from pathlib import Path

# Explicitly define the base directory using your exact current folder path
BASE_DIR = Path(r"C:\Users\fathi\OneDrive\Desktop\smartcare-noshow-ai\New folder")

# Absolute path definitions
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "smartcare_ai_dataset_1000.csv"
MODEL_PATH = BASE_DIR / "model"
DECISION_REPORT_PATH = BASE_DIR / "reports"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed"