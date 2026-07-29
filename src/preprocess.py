import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_and_preprocess_data(filepath='data/smartcare_ai_dataset_1000.csv'):
    df = pd.read_csv(filepath)
    
    # 1. Drop identifier and target-leakage columns
    leakage_cols = ['record_id', 'patient_id', 'appointment_status', 'no_show', 'readmitted_30_days', 'disease_risk_level']
    X = df.drop(columns=leakage_cols)
    y = df['no_show']
    
    # 2. Extract date features
    if 'appointment_date' in X.columns:
        X['appointment_date'] = pd.to_datetime(X['appointment_date'])
        X['app_month'] = X['appointment_date'].dt.month
        X['app_dayofweek'] = X['appointment_date'].dt.dayofweek
        X = X.drop(columns=['appointment_date'])
        
    # 3. Handle missing room_type values
    X['room_type'] = X['room_type'].fillna('None')
    
    # 4. Feature engineering: missed appointment ratio
    X['missed_ratio'] = X['missed_previous_appointments'] / (X['previous_appointments'] + 1)
    
    # Identify numerical and categorical columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Define ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, preprocessor, X