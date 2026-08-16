import os
import webbrowser
import threading
import time
import pandas as pd
from flask import Flask, request, jsonify, render_template
from model_loader import ( models, preprocessor, background_data)
from shapAI import explain_prediction
from utils.server_path import ( TEMPLATE_PATH, STATIC_PATH)


# Flask Configuration
app = Flask(
    __name__,
    template_folder=str(TEMPLATE_PATH),
    static_folder=str(STATIC_PATH)
)

# Flask app ui
@app.route("/")
def home():

    return render_template("index.html")

# Browser session tracker for the tab is runing or not
last_heartbeat = time.time()


@app.route("/heartbeat", methods=["POST"])
def heartbeat():

    global last_heartbeat
    last_heartbeat = time.time()
    return "", 204


def monitor_browser():

    global last_heartbeat
    while True:
        time.sleep(3)
        if time.time() - last_heartbeat > 10:

            print("Browser closed. Shutting down SmartCare...")

            os._exit(0)

# Prediction API route
@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.json

        model_name = data.pop(
            "model",
            "Logistic Regression"
        )

        if model_name not in models:

            return jsonify({
                "error": "Model not available"
            }), 400

        model = models[model_name]

        print(f"Using model : {model_name}")

        input_df = pd.DataFrame([data])

        # Feature Engineering
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

        input_df.drop(
            columns=["appointment_date"],
            inplace=True
        )

        # Preprocessing
        processed = preprocessor.transform(
            input_df
        )

        # Prediction
        probability = model.predict_proba(
            processed
        )[0][1]

        prediction = model.predict(
            processed
        )[0]

        # SHAP
        explanation = explain_prediction(
            model,
            model_name,
            processed,
            preprocessor.get_feature_names_out(),
            background_data
        )

        return jsonify({

            "model": model_name,
            "prediction": int(prediction),
            "probability": float(probability),
            "risk": (
                "High Risk"
                if probability >= 0.65
                else
                "Medium Risk"
                if probability >= 0.40
                else
                "Low Risk"
            ),
            "explanation": explanation
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

def open_browser():
    time.sleep(2)
    webbrowser.open(
        "http://127.0.0.1:5000"
    )

if __name__ == "__main__":

    print("Starting SmartCare Web Server...")

    threading.Thread(
        target=open_browser
    ).start()

    threading.Thread(
        target=monitor_browser,
        daemon=True
    ).start()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )