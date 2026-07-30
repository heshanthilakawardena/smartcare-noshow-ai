import pandas as pd
from pathlib import Path

def LoadData(file_path):

    file = Path(file_path)

    if not file.exists() and file.suffix.lower() != ".csv":
        raise FileNotFoundError(f"File not found: {file}")

    else: 

        print(f"Loading dataset: {file.name}")

    df = pd.read_csv(file, low_memory=False)

    print(f"Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")

    return df



