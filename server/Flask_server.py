import webbrowser
import threading
import time
import pandas as pd
from flask import Flask, request, jsonify, render_template
from model_loader import (
    models,
    preprocessor,
    background_data
)
from shapAI import explain_prediction
from utils.server_path import (
    TEMPLATE_PATH,
    STATIC_PATH
)


# ==================================
# Flask Configuration
# ==================================

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_PATH),
    static_folder=str(STATIC_PATH)
)


# ==================================
# Home Page
# ==================================

@app.route("/")
def home():

    return render_template("index.html")


# ==================================
# Prediction API
# ==================================

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
                "error": "Model not available"
            }), 400

        model = models[model_name]

        print(f"Using model : {model_name}")

        # -------------------------
        # Create DataFrame
        # -------------------------

        input_df = pd.DataFrame([data])

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

# ==================================
# Run Flask
# ==================================

if __name__ == "__main__":

    print("🚀 Starting SmartCare Web Server...")

    threading.Thread(
        target=open_browser
    ).start()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )