from flask import Flask,request,jsonify, redirect
import requests
import pandas as pd
from model_loader import (
    models,
    preprocessor
)
from shapAI import explain_prediction
import subprocess
import threading
import time
import webbrowser
from utils.server_path import (STEAMLIT)



app = Flask(__name__)



@app.route(
    "/predict",
    methods=["POST"]
)

def predict():


    try:

        data = request.json


        # -------------------------
        # Select Model
        # -------------------------

        model_name = data.pop(
            "model",
            "Logistic Regression"
        )


        if model_name not in models:

            return jsonify({

                "error":
                "Model not available"

            }),400



        model = models[model_name]



        print(
            f"Using model : {model_name}"
        )



        # -------------------------
        # Create DataFrame
        # -------------------------

        input_df = pd.DataFrame(
            [data]
        )



        # -------------------------
        # Feature Engineering
        # -------------------------

        input_df["appointment_date"] = pd.to_datetime(
            input_df["appointment_date"]
        )

        input_df["appointment_year"] = (
            input_df["appointment_date"].dt.year
        )

        input_df["appointment_month"] = (
            input_df["appointment_date"].dt.month
        )

        input_df["appointment_day"] = (
            input_df["appointment_date"].dt.day
        )

        input_df["appointment_dayofweek"] = (
            input_df["appointment_date"].dt.dayofweek
        )

        input_df["appointment_weekend"] = (
            input_df["appointment_dayofweek"] >= 5
        ).astype(int)


        # Remove original date column if it was dropped during training
        input_df.drop(
            columns=["appointment_date"],
            inplace=True
        )


        # -------------------------
        # Preprocessing
        # -------------------------

        processed = preprocessor.transform(
            input_df
        )



        # -------------------------
        # Prediction
        # -------------------------

        probability = model.predict_proba(
            processed
        )[0][1]


        prediction = model.predict(
            processed
        )[0]



        # -------------------------
        # SHAP
        # -------------------------

        explanation = explain_prediction(

            model,

            model_name,

            processed,

            preprocessor.get_feature_names_out()

        )



        return jsonify({

            "model":
            model_name,


            "prediction":
            int(prediction),


            "probability":
            float(probability),


            "risk":

            (
                "High Risk"
                if probability >=0.65

                else

                "Medium Risk"
                if probability >=0.40

                else

                "Low Risk"
            ),


            "explanation":
            explanation

        })



    except Exception as e:


        return jsonify({

            "error":
            str(e)

        }),500

# ==================================
# Home Redirect
# ==================================

@app.route("/")
def home():

    return redirect(
        "http://localhost:8501"
    )


# ==================================
# Start Streamlit Automatically
# ==================================

def start_streamlit():

    print(f"Starting Streamlit from: {STEAMLIT}")

    subprocess.Popen(
        [
            "streamlit",
            "run",
            str(STEAMLIT)
        ]
    )

    time.sleep(5)
    webbrowser.open("http://localhost:8501")



if __name__=="__main__":


    print(
        "🚀 Starting SmartCare Streamlit UI..."
    )


    threading.Thread(
        target=start_streamlit,
        daemon=True
    ).start()


    print(
        "🚀 Starting Flask API Server..."
    )


    app.run(
        host="127.0.0.1",
        port=5000
    )