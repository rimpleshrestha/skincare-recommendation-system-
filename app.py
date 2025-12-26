# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import os

app = Flask(__name__)
CORS(app)
# ------------------------------
# Constants
# ------------------------------
MODELS_FOLDER = "models"

FEATURE_COLUMNS = [
    "age", "skin_type", "skin_tone", "primary_concern", "secondary_concerns",
    "occupation", "sleep_hours", "daily_water_intake", "stress_level",
    "exercise_frequency", "diet_type", "junk_food_frequency", "dairy_intake",
    "sugar_intake", "sun_exposure", "sunscreen_usage", "allergies"
]

CATEGORICAL_COLUMNS = [
    "skin_type", "skin_tone", "primary_concern", "secondary_concerns",
    "occupation", "diet_type", "sun_exposure", "sunscreen_usage", "allergies"
]

NUMERIC_COLUMNS = [
    "age", "sleep_hours", "daily_water_intake", "stress_level",
    "exercise_frequency", "junk_food_frequency", "dairy_intake", "sugar_intake"
]

# ------------------------------
# Load encoders
# ------------------------------
with open(os.path.join(MODELS_FOLDER, "feature_encoders.pkl"), "rb") as f:
    feature_encoders = pickle.load(f)

# ------------------------------
# Load models and target encoders
# ------------------------------
models_info = {
    "moisturizer": ("moisturizer_model.txt", "moisturizer_target_encoder.pkl"),
    "serum": ("serum_model.txt", "serum_target_encoder.pkl"),
    "sunscreen": ("sunscreen_model.txt", "sunscreen_target_encoder.pkl"),
    "treatment": ("treatment_model.txt", "treatment_target_encoder.pkl"),
    "avoid_ingredients": ("avoid_ingredients_model.txt", "avoid_ingredients_target_encoder.pkl")
}

models = {}
target_encoders = {}

for key, (model_file, encoder_file) in models_info.items():
    models[key] = lgb.Booster(model_file=os.path.join(MODELS_FOLDER, model_file))
    with open(os.path.join(MODELS_FOLDER, encoder_file), "rb") as f:
        target_encoders[key] = pickle.load(f)

# ------------------------------
# Skin type description
# ------------------------------
SKIN_TYPE_DESC = {
    "Oily": "Oily skin tends to produce excess sebum, which can lead to acne, enlarged pores, and occasional dullness.",
    "Dry": "Dry skin often feels tight, flaky, and may lack natural glow.",
    "Combination": "Combination skin is oily in some areas and dry in others, requiring balanced care.",
    "Normal": "Normal skin is well-balanced, with few concerns.",
    "Sensitive": "Sensitive skin is prone to irritation and redness and requires gentle products."
}

# ------------------------------
# Helper function to process input
# ------------------------------
def process_input(user_input: dict) -> pd.DataFrame:
    """Convert user input JSON to DataFrame ready for prediction"""
    X_user = pd.DataFrame([user_input])

    # Convert numeric fields
    for col in NUMERIC_COLUMNS:
        X_user[col] = pd.to_numeric(X_user[col], errors="coerce").fillna(0)

    # Encode categorical fields
    for col in CATEGORICAL_COLUMNS:
        le = feature_encoders[col]
        if col in X_user:
            X_user[col] = le.transform(X_user[col].astype(str))
        else:
            X_user[col] = 0  # default if missing

    return X_user

# ------------------------------
# Prediction endpoint
# ------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_input = request.json
        X_user = process_input(user_input)

        response = {}
        # Start paragraph with skin type explanation
        skin_type = user_input.get("skin_type", "Normal")
        skin_tone = user_input.get("skin_tone", "")
        primary = user_input.get("primary_concern", "")
        secondary = user_input.get("secondary_concerns", "")
        
        advice_text = f"Based on your quiz answers, your skin type is {skin_type} with a {skin_tone} tone. "
        advice_text += SKIN_TYPE_DESC.get(skin_type, "")
        advice_text += f" Your primary concern is {primary}, and secondary concerns include {secondary}. "
        advice_text += "Maintaining proper hydration, using suitable products, and protecting your skin from sun exposure are important.\n\n"
        advice_text += "Recommended products for your skin:\n"

        recommendations = {}

        for key in models.keys():
            model = models[key]
            target_encoder = target_encoders[key]

            preds = model.predict(X_user)
            if preds.ndim > 1:
                # Multi-class: get top 3 recommendations
                top_indices = np.argsort(preds[0])[::-1][:3]
                recs = [target_encoder.inverse_transform([i])[0] for i in top_indices]
            else:
                recs = [target_encoder.inverse_transform([int(np.argmax(preds))])[0]]

            recommendations[key] = recs
            advice_text += f"- {key.capitalize()}: {', '.join(recs)}\n"

        response["recommendations"] = recommendations
        response["advice"] = advice_text

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ------------------------------
# Run the app
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
