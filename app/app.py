from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("../model/model.pkl")
le = joblib.load("../model/label_encoder.pkl")

# Load CSV
treatment_df = pd.read_csv("../data/treatment_prevention.csv")
treatment_df["Disease"] = treatment_df["Disease"].str.strip()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    form_data = {
        "fever": "",
        "cough": "",
        "fatigue": "",
        "breathing": "",
        "gender": "",
        "bp": "",
        "chol": ""
    }
    if request.method == "POST":

        form_data["fever"] = request.form["fever"]
        form_data["cough"] = request.form["cough"]
        form_data["fatigue"] = request.form["fatigue"]
        form_data["breathing"] = request.form["breathing"]
        form_data["gender"] = request.form["gender"]
        form_data["bp"] = request.form["bp"]
        form_data["chol"] = request.form["chol"]

        input_data = [
            int(form_data["fever"]),
            int(form_data["cough"]),
            int(form_data["fatigue"]),
            int(form_data["breathing"]),
            int(form_data["gender"]),
            int(form_data["bp"]),
            int(form_data["chol"])
        ]

        # Prediction
        pred = model.predict([input_data])[0]
        prediction = le.inverse_transform([pred])[0].strip()

        # Find disease info from CSV
        disease_info = treatment_df[
            treatment_df["Disease"].str.lower() == prediction.lower()
        ]

        if not disease_info.empty:
            disease_info = disease_info.iloc[0]

            result = {
                "disease": prediction,
                "category": disease_info["Category"],
                "diet": disease_info["Diet/Foods"],
                "medicines": disease_info["Medicines"],
                "prevention": disease_info["Prevention"]
            }
        else:
            result = {
                "disease": prediction,
                "category": "Not found",
                "diet": "Not found",
                "medicines": "Not found",
                "prevention": "Not found"
            }

    # 
    # ALWAYS return index.html
    return render_template("index.html", result=result, form_data=form_data)


if __name__ == "__main__":
    app.run(debug=True)