import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle

# ------------------------------
# Columns setup
# ------------------------------
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
# Load feature encoders
# ------------------------------
with open("feature_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

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
    models[key] = lgb.Booster(model_file=model_file)
    with open(encoder_file, "rb") as f:
        target_encoders[key] = pickle.load(f)

# ------------------------------
# Ask user for input interactively
# ------------------------------
user_input = {}

# Numeric inputs
for col in NUMERIC_COLUMNS:
    ans = input(f"{col.replace('_', ' ').capitalize()} (enter number or range like '6-8', default 0): ")
    user_input[col] = ans if ans else 0

# Categorical inputs
for col in CATEGORICAL_COLUMNS:
    valid_options = list(encoders[col].classes_)
    while True:
        ans = input(f"{col.replace('_', ' ').capitalize()} (options: {valid_options}): ")
        if ans in valid_options:
            user_input[col] = ans
            break
        print(f"Invalid option. Choose from {valid_options}")

# ------------------------------
# Convert user input to DataFrame
# ------------------------------
X_user = pd.DataFrame([user_input])

# Convert numeric fields
for col in NUMERIC_COLUMNS:
    X_user[col] = pd.to_numeric(X_user[col], errors="coerce").fillna(0)

# Encode categorical features
for col in CATEGORICAL_COLUMNS:
    le = encoders[col]
    X_user[col] = le.transform(X_user[col].astype(str))

# ------------------------------
# Predict for each model
# ------------------------------
for key in models.keys():
    model = models[key]
    target_encoder = target_encoders[key]

    preds = model.predict(X_user)
    if preds.ndim > 1:
        # Multi-class: get top 3 recommendations
        top_indices = np.argsort(preds[0])[::-1][:3]
        recommendations = [target_encoder.inverse_transform([i])[0] for i in top_indices]
    else:
        recommendations = [target_encoder.inverse_transform([int(np.argmax(preds))])[0]]

    print(f"\nTop recommendations for {key}: {recommendations}")
