# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import lightgbm as lgb
import pickle
import os

# ------------------------------
# FastAPI initialization
# ------------------------------
app = FastAPI(title="Skincare Recommendation API")

# ------------------------------
# Paths
# ------------------------------
MODEL_DIR = "models"

# ------------------------------
# Load LightGBM models
# ------------------------------
models = {
    "serum": lgb.Booster(model_file=os.path.join(MODEL_DIR, "serum_model.txt")),
    "moisturizer": lgb.Booster(model_file=os.path.join(MODEL_DIR, "moisturizer_model.txt")),
    "sunscreen": lgb.Booster(model_file=os.path.join(MODEL_DIR, "sunscreen_model.txt")),
    "treatment": lgb.Booster(model_file=os.path.join(MODEL_DIR, "treatment_model.txt")),
    "avoid_ingredients": lgb.Booster(model_file=os.path.join(MODEL_DIR, "avoid_ingredients_model.txt")),
}

# ------------------------------
# Load target encoders
# ------------------------------
target_encoders = {
    "serum": pickle.load(open(os.path.join(MODEL_DIR, "serum_target_encoder.pkl"), "rb")),
    "moisturizer": pickle.load(open(os.path.join(MODEL_DIR, "moisturizer_target_encoder.pkl"), "rb")),
    "sunscreen": pickle.load(open(os.path.join(MODEL_DIR, "sunscreen_target_encoder.pkl"), "rb")),
    "treatment": pickle.load(open(os.path.join(MODEL_DIR, "treatment_target_encoder.pkl"), "rb")),
    "avoid_ingredients": pickle.load(open(os.path.join(MODEL_DIR, "avoid_ingredients_target_encoder.pkl"), "rb")),
}

# ------------------------------
# Load feature encoders for categorical columns
# ------------------------------
encoders = pickle.load(open(os.path.join(MODEL_DIR, "feature_encoders.pkl"), "rb"))

# ------------------------------
# Feature columns (must match training)
# ------------------------------
FEATURE_COLUMNS = [
    "age", "skin_type", "skin_tone", "primary_concern", "secondary_concerns",
    "occupation", "sleep_hours", "daily_water_intake", "stress_level",
    "exercise_frequency", "diet_type", "junk_food_frequency", "dairy_intake",
    "sugar_intake", "sun_exposure", "sunscreen_usage", "allergies"
]

# ------------------------------
# Input schema
# ------------------------------
class SkinInput(BaseModel):
    age: int
    skin_type: str
    skin_tone: str
    primary_concern: str
    secondary_concerns: str
    occupation: str
    sleep_hours: float
    daily_water_intake: float
    stress_level: float
    exercise_frequency: float
    diet_type: str
    junk_food_frequency: float
    dairy_intake: float
    sugar_intake: float
    sun_exposure: str
    sunscreen_usage: str
    allergies: str

# ------------------------------
# Prediction helper
# ------------------------------
def predict(model_name, input_data: SkinInput):
    df = pd.DataFrame([input_data.dict()])

    # Encode categorical features
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))

    # Ensure column order
    df = df[FEATURE_COLUMNS]

    # Predict
    model = models[model_name]
    pred_probs = model.predict(df)
    pred_index = pred_probs.argmax(axis=1)[0]

    # Decode label
    recommendation = target_encoders[model_name].inverse_transform([pred_index])[0]
    return recommendation

# ------------------------------
# Endpoints
# ------------------------------
@app.post("/predict-serum")
def predict_serum(input: SkinInput):
    rec = predict("serum", input)
    return {"recommendation": rec}

@app.post("/predict-moisturizer")
def predict_moisturizer(input: SkinInput):
    rec = predict("moisturizer", input)
    return {"recommendation": rec}

@app.post("/predict-sunscreen")
def predict_sunscreen(input: SkinInput):
    rec = predict("sunscreen", input)
    return {"recommendation": rec}

@app.post("/predict-treatment")
def predict_treatment(input: SkinInput):
    rec = predict("treatment", input)
    return {"recommendation": rec}

@app.post("/predict-avoid-ingredients")
def predict_avoid_ingredients(input: SkinInput):
    rec = predict("avoid_ingredients", input)
    return {"recommendation": rec}

# ------------------------------
# Optional root endpoint
# ------------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to the Skincare Recommendation API"}
