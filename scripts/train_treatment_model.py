import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle

# ------------------------------
# File & target setup
# ------------------------------
PARQUET_FILE = "cleaned_skincare_dataset.parquet"
TARGET = "rec_treatment"  # Treatment target

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

# ------------------------------
# Load dataset
# ------------------------------
df = pd.read_parquet(PARQUET_FILE, engine="pyarrow")

# Convert numeric-like object columns to float
numeric_cols = ["sleep_hours", "daily_water_intake", "stress_level", 
                "exercise_frequency", "junk_food_frequency", "dairy_intake", "sugar_intake"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# Label encode categorical features
encoders = {}
for col in CATEGORICAL_COLUMNS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ------------------------------
# Encode target for multi-class
# ------------------------------
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df[TARGET].astype(str))
X = df[FEATURE_COLUMNS]

# Save target encoder
with open("treatment_target_encoder.pkl", "wb") as f:
    pickle.dump(target_encoder, f)

# ------------------------------
# Prepare LightGBM dataset
# ------------------------------
train_data = lgb.Dataset(X, label=y, free_raw_data=False)

# ------------------------------
# Train LightGBM model
# ------------------------------
model = lgb.train(
    params={
        "objective": "multiclass",
        "num_class": len(np.unique(y)),
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1
    },
    train_set=train_data,
    num_boost_round=120
)

# ------------------------------
# Save the trained model
# ------------------------------
model.save_model("treatment_model.txt")
print("✅ Treatment model trained and saved as treatment_model.txt")
