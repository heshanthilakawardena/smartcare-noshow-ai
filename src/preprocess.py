import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies domain-specific feature engineering to the raw dataset.
    """
    df = df.copy()

    # 1. Total Financial Burden
    financial_cols = ['consultation_fee_lkr', 'room_charge_lkr', 'lab_charge_lkr', 'medicine_charge_lkr']
    for col in financial_cols:
        if col not in df.columns:
            df[col] = 0
    df['total_financial_burden'] = df[financial_cols].sum(axis=1)

    # 2. Chronic Patient Flag
    if 'age' in df.columns and 'diagnosis' in df.columns:
        df['chronic_patient'] = (
            (df['age'] > 60) | 
            (df['diagnosis'].astype(str).str.strip().isin(['Diabetes', 'Hypertension', 'Heart Disease', 'Asthma']))
        ).astype(int)

    # 3. High Missed Appointments Ratio Flag
    if 'previous_appointments' in df.columns and 'missed_previous_appointments' in df.columns:
        prev_appts = df['previous_appointments'].replace(0, 1) # Avoid division by zero
        df['missed_ratio'] = df['missed_previous_appointments'] / prev_appts
        df['high_missed_ratio_flag'] = (df['missed_ratio'] > 0.3).astype(int)

    # 4. Long Wait Flag (> 14 days)
    if 'waiting_days' in df.columns:
        df['long_wait_flag'] = (df['waiting_days'] > 14).astype(int)

    return df

def apply_iqr_clipping(df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
    """
    Clips numeric outliers using the Interquartile Range (IQR) method.
    """
    df_clipped = df.copy()
    for col in num_cols:
        if col in df_clipped.columns:
            df_clipped[col] = pd.to_numeric(df_clipped[col], errors='coerce')
            Q1 = df_clipped[col].quantile(0.25)
            Q3 = df_clipped[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clipped[col] = np.clip(df_clipped[col], lower_bound, upper_bound)
    return df_clipped

def load_and_preprocess_data(csv_path: str):
    """
    Loads raw CSV data, cleans duplicates, splits target y and features X,
    engineers features, and creates an Scikit-Learn preprocessing pipeline.
    """
    df = pd.read_csv(csv_path)

    # Clean duplicates
    initial_len = len(df)
    df = df.drop_duplicates()
    print(f"Duplicates detected and removed: {initial_len - len(df)}")

    # Target Column Extraction
    target_col = 'no_show'
    if target_col in df.columns:
        y = df[target_col].astype(int)
        X = df.drop(columns=[target_col])
    else:
        raise ValueError(f"Target column '{target_col}' not found in dataset!")

    # Drop non-predictive IDs AND leakage columns (like appointment_status)
    drop_cols = ['patient_id', 'record_id', 'appointment_date', 'appointment_status', 'status']
    X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors='ignore')

    # Apply Feature Engineering
    X = engineer_features(X)

    # Separate Numerical vs Categorical Columns strictly by data type
    num_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Apply IQR Outlier Clipping
    X[num_cols] = apply_iqr_clipping(X[num_cols], num_cols)

    # Sklearn Transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor, list(X.columns)