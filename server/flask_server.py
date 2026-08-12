import os
import webbrowser
import threading
import time
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from model_loader import (
    models,
    preprocessor,
    background_data,
    clean_to_float_array
)
from shapAI import explain_prediction
from utils.server_path import (
    TEMPLATE_PATH,
    STATIC_PATH
)


#Flask Configuration


app = Flask(
    __name__,
    template_folder=str(TEMPLATE_PATH),
    static_folder=str(STATIC_PATH)
)

# Enable CORS for cross-origin frontend communication
CORS(app)


#Home Page


@app.route("/")
def home():
    return render_template("index.html")


#Prediction API


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Select Model
        model_name = data.pop("model", "Logistic Regression")

        if model_name not in models:
            return jsonify({"error": "Model not available"}), 400

        model = models[model_name]
        print(f"Using model : {model_name}")

        # Create DataFrame & Feature Engineering
        input_df = pd.DataFrame([data])

        input_df["appointment_date"] = pd.to_datetime(input_df["appointment_date"])
        input_df["appointment_year"] = input_df["appointment_date"].dt.year
        input_df["appointment_month"] = input_df["appointment_date"].dt.month
        input_df["appointment_day"] = input_df["appointment_date"].dt.day
        input_df["appointment_dayofweek"] = input_df["appointment_date"].dt.dayofweek
        input_df["appointment_weekend"] = (input_df["appointment_dayofweek"] >= 5).astype(int)
        input_df.drop(columns=["appointment_date"], inplace=True)

        # Preprocessing & Force Conversion to Float Array
        raw_processed = preprocessor.transform(input_df)
        
        # CLEAN PROCESSED INPUT BEFORE CALLING ANY MODEL METHOD
        processed = clean_to_float_array(raw_processed)

        # Print debug type info to terminal
        print(f"Processed shape: {processed.shape}, dtype: {processed.dtype}")

        # Prediction
        probability = model.predict_proba(processed)[0][1]
        prediction = model.predict(processed)[0]

        # SHAP Explanation
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
                "High Risk" if probability >= 0.65
                else "Medium Risk" if probability >= 0.40
                else "Low Risk"
            ),
            "explanation": explanation
        })

    except Exception as e:
        print(f"Error executing prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def open_browser(port):
    time.sleep(2)
    webbrowser.open(f"http://localhost:{port}")



# Local Development Runner


if __name__ == "__main__":
    print("Starting SmartCare Web Server locally...")
    port = int(os.environ.get("PORT", 8080))
    threading.Thread(target=open_browser, args=(port,)).start()

    app.run(
        host="127.0.0.1",
        port=port,
        debug=False,
        use_reloader=False
    )