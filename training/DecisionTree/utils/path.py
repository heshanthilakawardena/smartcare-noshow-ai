from pathlib import Path

# From training/DecisionTree/utils/path.py:
# .parent -> utils
# .parents[1] -> DecisionTree
# .parents[2] -> training
# .parents[3] -> smartcare-noshow-ai (Root)
BAS_DIR = Path(__file__).resolve().parents[3]

RAW_DATA_PATH = BAS_DIR / "data" / "raw" / "smartcare_ai_dataset_1000.csv"
MODEL_PATH = BAS_DIR / "training" / "DecisionTree" / "model"
DECISION_REPORT_PATH = BAS_DIR / "training" / "DecisionTree" / "reports"
PROCESSED_DATA_PATH = BAS_DIR / "training" / "DecisionTree" / "data" / "processed"