import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="SmartCare AI - No-Show Predictor", layout="wide")

st.title("🏥 SmartCare Hospital - Appointment No-Show AI Decision Support")

# Model Selection Sidebar
st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox(
    "Choose AI Model for Prediction:",
    ["XGBoost", "Random Forest"]
)

# Load Selected Model Pipeline
@st.cache_resource
def load_model(model_name):
    if model_name == "XGBoost":
        return joblib.load("models/xgboost_model.joblib")
    elif model_name == "Random Forest":
        return joblib.load("models/random_forest_model.joblib")

pipeline = load_model(selected_model_name)

st.sidebar.markdown("---")
st.sidebar.header("Patient & Appointment Details")

# Input fields
age = st.sidebar.slider("Age", 0, 100, 40)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
blood_group = st.sidebar.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"])
department = st.sidebar.selectbox("Department", ["General Medicine", "Cardiology", "Neurology", "Orthopedics"])
diagnosis = st.sidebar.selectbox("Diagnosis", ["Migraine", "Diabetes", "Back Pain", "Asthma", "Hypertension"])
waiting_days = st.sidebar.number_input("Waiting Days", 0, 60, 10)
previous_appointments = st.sidebar.number_input("Previous Appointments Count", 0, 20, 2)
missed_previous_appointments = st.sidebar.number_input("Missed Previous Appointments", 0, 10, 0)
admitted = st.sidebar.selectbox("Admitted Previously", [0, 1])
bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 25.0)

if st.button(f"Predict No-Show using {selected_model_name}"):
    input_data = pd.DataFrame([{
        'age': age,
        'gender': gender,
        'blood_group': blood_group,
        'department': department,
        'diagnosis': diagnosis,
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
        'app_month': 5,
        'app_dayofweek': 2,
        'missed_ratio': missed_previous_appointments / (previous_appointments + 1)
    }])
    
    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1]
    
    st.subheader(f"Prediction Result ({selected_model_name})")
    if prediction == 1:
        st.error(f"🚨 High Risk of No-Show! (Probability: {probability:.2%})")
        st.write("**Recommendation:** Send SMS reminder, make a direct phone call, or offer double-booking.")
    else:
        st.success(f"✅ Likely to Attend (Probability of No-Show: {probability:.2%})")