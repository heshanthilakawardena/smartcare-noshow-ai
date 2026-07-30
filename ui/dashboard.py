import streamlit as st
import requests
from datetime import date


# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="SmartCare AI",
    layout="wide"
)


st.title(
    "🏥 SmartCare Hospital AI Appointment No-Show Predictor"
)


st.write(
    "Predict patient appointment no-show risk using Logistic Regression and KNN models."
)



# ==========================
# Flask API
# ==========================

API_URL = "http://localhost:5000/predict"



# ==========================
# Model Selection
# ==========================

st.sidebar.header(
    "🤖 AI Model Selection"
)


selected_model = st.sidebar.selectbox(

    "Select Model",

    [
        "Logistic Regression",
        "KNN"
    ]

)



# ==========================
# Patient Information
# ==========================

st.sidebar.header(
    "👤 Patient Information"
)



record_id = st.sidebar.number_input(
    "Record ID",
    1,
    100000,
    1
)


age = st.sidebar.number_input(
    "Age",
    0,
    100,
    40
)



gender = st.sidebar.selectbox(

    "Gender",

    [
        "Male",
        "Female"
    ]

)



blood_group = st.sidebar.selectbox(

    "Blood Group",

    [
        "A+",
        "A-",
        "B+",
        "B-",
        "O+",
        "O-",
        "AB+",
        "AB-"
    ]

)



# ==========================
# Appointment Information
# ==========================


st.sidebar.header(
    "📅 Appointment Details"
)



department = st.sidebar.selectbox(

    "Department",

    [
        "Cardiology",
        "Neurology",
        "General Medicine",
        "Orthopedics",
        "Dermatology",
        "ENT",
        "Pediatrics"
    ]

)



diagnosis = st.sidebar.selectbox(

    "Diagnosis",

    [
        "Diabetes",
        "Hypertension",
        "Migraine",
        "Asthma",
        "Back Pain"
    ]

)



appointment_date = st.sidebar.date_input(

    "Appointment Date",

    date.today()

)



waiting_days = st.sidebar.number_input(

    "Waiting Days",

    0,
    365,
    5

)



previous_appointments = st.sidebar.number_input(

    "Previous Appointments",

    0,
    100,
    2

)



missed_previous_appointments = st.sidebar.number_input(

    "Missed Previous Appointments",

    0,
    100,
    0

)



appointment_status = st.sidebar.selectbox(

    "Appointment Status",

    [
        "Scheduled",
        "Completed",
        "Cancelled",
        "No-Show"
    ]

)



# ==========================
# Medical Information
# ==========================


st.sidebar.header(
    "🩺 Medical Information"
)



admitted = st.sidebar.selectbox(

    "Admitted",

    [
        0,
        1
    ]

)



room_type = st.sidebar.selectbox(

    "Room Type",

    [
        "General",
        "Private",
        "ICU"
    ]

)



length_of_stay_days = st.sidebar.number_input(

    "Length Of Stay Days",

    0,
    365,
    0

)



previous_admissions = st.sidebar.number_input(

    "Previous Admissions",

    0,
    100,
    0

)



systolic_bp = st.sidebar.number_input(

    "Systolic BP",

    50,
    250,
    120

)



diastolic_bp = st.sidebar.number_input(

    "Diastolic BP",

    30,
    150,
    80

)



blood_sugar_mg_dl = st.sidebar.number_input(

    "Blood Sugar mg/dL",

    50,
    500,
    100

)



cholesterol_mg_dl = st.sidebar.number_input(

    "Cholesterol mg/dL",

    50,
    500,
    180

)



bmi = st.sidebar.number_input(

    "BMI",

    10.0,
    60.0,
    25.0

)



lab_tests_count = st.sidebar.number_input(

    "Lab Tests Count",

    0,
    100,
    1

)



treatments_count = st.sidebar.number_input(

    "Treatments Count",

    0,
    100,
    1

)



# ==========================
# Billing
# ==========================


st.sidebar.header(
    "💳 Billing Information"
)



consultation_fee_lkr = st.sidebar.number_input(

    "Consultation Fee",

    0,
    500000,
    2000

)



room_charge_lkr = st.sidebar.number_input(

    "Room Charge",

    0,
    500000,
    0

)



lab_charge_lkr = st.sidebar.number_input(

    "Lab Charge",

    0,
    500000,
    0

)



medicine_charge_lkr = st.sidebar.number_input(

    "Medicine Charge",

    0,
    500000,
    1000

)



total_bill_lkr = st.sidebar.number_input(

    "Total Bill",

    0,
    1000000,
    3000

)



payment_status = st.sidebar.selectbox(

    "Payment Status",

    [
        "Paid",
        "Partially Paid",
        "Unpaid"
    ]

)



payment_method = st.sidebar.selectbox(

    "Payment Method",

    [
        "Cash",
        "Card",
        "Insurance"
    ]

)



# ==========================
# Prediction Request
# ==========================


if st.button(
    "🚀 Predict No-Show Risk"
):


    payload = {


        "model": selected_model,


        "record_id": record_id,

        "age": age,

        "gender": gender,

        "blood_group": blood_group,

        "department": department,

        "diagnosis": diagnosis,


        "appointment_date":
            str(appointment_date),


        "waiting_days": waiting_days,


        "previous_appointments":
            previous_appointments,


        "missed_previous_appointments":
            missed_previous_appointments,


        "appointment_status":
            appointment_status,


        "admitted": admitted,


        "room_type": room_type,


        "length_of_stay_days":
            length_of_stay_days,


        "previous_admissions":
            previous_admissions,


        "systolic_bp": systolic_bp,


        "diastolic_bp": diastolic_bp,


        "blood_sugar_mg_dl":
            blood_sugar_mg_dl,


        "cholesterol_mg_dl":
            cholesterol_mg_dl,


        "bmi": bmi,


        "lab_tests_count":
            lab_tests_count,


        "treatments_count":
            treatments_count,


        "consultation_fee_lkr":
            consultation_fee_lkr,


        "room_charge_lkr":
            room_charge_lkr,


        "lab_charge_lkr":
            lab_charge_lkr,


        "medicine_charge_lkr":
            medicine_charge_lkr,


        "total_bill_lkr":
            total_bill_lkr,


        "payment_status":
            payment_status,


        "payment_method":
            payment_method

    }



    try:

        response = requests.post(

            API_URL,

            json=payload,

            timeout=30

        )


        result = response.json()



        if "error" in result:

            st.error(
                result["error"]
            )


        else:


            probability = result["probability"]



            st.subheader(
                "Prediction Result"
            )



            st.info(
                f"Model Used: {result['model']}"
            )


            st.metric(

                "No-Show Probability",

                f"{probability:.2%}"

            )



            if result["risk"] == "High Risk":

                st.error(
                    "🚨 High Risk of No-Show"
                )


            elif result["risk"] == "Medium Risk":

                st.warning(
                    "⚠️ Medium Risk of No-Show"
                )


            else:

                st.success(
                    "✅ Low Risk - Patient likely to attend"
                )



            st.subheader(
                "🔍 SHAP Explainable AI Result"
            )


            st.json(
                result["explanation"]
            )



    except requests.exceptions.ConnectionError:


        st.error(
            "❌ Flask server is not running."
        )


    except Exception as e:


        st.error(
            f"Error : {e}"
        )