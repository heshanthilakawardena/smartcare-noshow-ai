from pathlib import Path

BAS_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = BAS_DIR / "data" / "raw"/ "smartcare_ai_dataset_1000.txt"
MODEL_PATH = BAS_DIR / "models"
REPORT_PATH = BAS_DIR / "reports"
PROCESSED_DATA_PATH = BAS_DIR / "data" / "processed"