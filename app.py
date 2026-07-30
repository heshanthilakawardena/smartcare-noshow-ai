import streamlit as st
import pandas as pd
import joblib
from src.preprocess import engineer_features

st.set_page_config(page_title="SmartCare AI - No-Show Predictor", layout="wide")

st.title("🏥 SmartCare Hospital - Appointment No-Show AI Decision Support")
st.write("Predict patient appointment attendance using trained machine learning models.")

# --- Dynamically Load Dropdown Categories from Dataset ---
@st.cache_data
def load_dataset_categories():
    df = pd.read_csv("data/smartcare_ai_dataset_1000.csv")
    
    diagnoses = sorted(df['diagnosis'].dropna().unique().tolist())
    departments = sorted(df['department'].dropna().unique().tolist())
    blood_groups = sorted(df['blood_group'].dropna().unique().tolist())
    
    return diagnoses, departments, blood_groups

try:
    diagnosis_options, department_options, blood_group_options = load_dataset_categories()
except Exception as e:
    # Fallback lists in case dataset path isn't found
    diagnosis_options = ["Migraine", "Diabetes", "Back Pain", "Asthma", "Hypertension"]
    department_options = ["General Medicine", "Cardiology", "Neurology", "Orthopedics"]
    blood_group_options = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]

# Model Selection Sidebar
st.sidebar.header("Model Selection")

model_options = [
    "XGBoost (Heshan)", 
    "Random Forest (Heshan)", 
    "Decision Tree (Zharah)",
    # "Logistic Regression (Binara)",
    # "KNN (Binara)"
]

selected_model_name = st.sidebar.selectbox(
    "Choose AI Model for Prediction:",
    model_options
)

MODEL_PATHS = {
    "XGBoost (Heshan)": "models/xgboost_model.joblib",
    "Random Forest (Heshan)": "models/random_forest_model.joblib",
    "Decision Tree (Zharah)": "models/decision_tree_zahra_model.joblib",
    # "Logistic Regression (Binara)": "models/logistic_regression_binara_model.joblib",
    # "KNN (Binara)": "models/knn_binara_model.joblib"
}

@st.cache_resource
def load_model(model_key):
    return joblib.load(MODEL_PATHS[model_key])

pipeline = load_model(selected_model_name)

st.sidebar.markdown("---")
st.sidebar.header("Patient & Appointment Details")

# Demographic & Clinical Inputs
age = st.sidebar.slider("Age", 0, 100, 40)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
blood_group = st.sidebar.selectbox("Blood Group", blood_group_options)
department = st.sidebar.selectbox("Department", department_options)
diagnosis = st.sidebar.selectbox("Diagnosis", diagnosis_options)
bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 25.0)

st.sidebar.markdown("---")
st.sidebar.header("Appointment History")

waiting_days = st.sidebar.number_input("Waiting Days", 0, 60, 10)
previous_appointments = st.sidebar.number_input("Previous Appointments Count", 0, 20, 2)
missed_previous_appointments = st.sidebar.number_input("Missed Previous Appointments", 0, 10, 0)
admitted = st.sidebar.selectbox("Admitted Previously", [0, 1])

# Automatically extract current appointment date
today = pd.Timestamp.now()
appointment_date_str = today.strftime("%Y-%m-%d")

if st.button(f"Predict No-Show using {selected_model_name}"):
    # 1. Build initial raw input dataframe with all dataset schema columns
    raw_input_data = pd.DataFrame([{
        'patient_id': 'P-TEMP',
        'record_id': 'R-TEMP',
        'age': age,
        'gender': gender,
        'blood_group': blood_group,
        'department': department,
        'diagnosis': diagnosis,
        'appointment_date': appointment_date_str,
        'waiting_days': waiting_days,
        'previous_appointments': previous_appointments,
        'missed_previous_appointments': missed_previous_appointments,
        'admitted': admitted,
        'room_type': 'None',
        'length_of_stay_days': 0,
        'previous_admissions': 0,
        'systolic_bp': 120,
        'diastolic_bp': 80,
        'blood_sugar_mg_dl': 100,
        'cholesterol_mg_dl': 180,
        'bmi': bmi,
        'lab_tests_count': 1,
        'treatments_count': 1,
        'consultation_fee_lkr': 2000,
        'room_charge_lkr': 0,
        'lab_charge_lkr': 0,
        'medicine_charge_lkr': 1000,
        'total_bill_lkr': 3000,
        'payment_status': 'Paid',
        'payment_method': 'Cash',
        'readmitted_30_days': 0,
        'disease_risk_level': 'Low'
    }])
    
    # 2. Run through the feature engineering pipeline
    processed_input_data = engineer_features(raw_input_data)
    
    # 3. Predict probability
    probability = pipeline.predict_proba(processed_input_data)[0][1]
    
    st.subheader(f"Prediction Result ({selected_model_name})")
    st.metric(label="Calculated No-Show Risk", value=f"{probability:.2%}")
    
    # Tiered Risk Logic
    if probability >= 0.65:
        st.error("🚨 High Risk of No-Show!")
        st.write("**Recommendation:** Schedule a direct phone call from clinic staff, send urgent automated SMS, or trigger double-booking.")
    elif probability >= 0.40:
        st.warning("⚠️ Moderate Risk of No-Show")
        st.write("**Recommendation:** Send standard automated SMS reminder 24 hours prior.")
    else:
        st.success("✅ Low Risk / Likely to Attend")
        st.write("**Recommendation:** Send standard automated appointment confirmation.")