from flask import Flask,request,jsonify
import pandas as pd
from model_loader import (
    models,
    preprocessor
)


from shap import explain_prediction



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





if __name__=="__main__":

    app.run(

        host="0.0.0.0",

        port=5000

    )