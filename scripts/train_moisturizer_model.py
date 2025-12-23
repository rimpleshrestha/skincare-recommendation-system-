import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle

# ---------------------------
# File & target
# ---------------------------
PARQUET_FILE = "cleaned_skincare_dataset.parquet"
TARGET = "rec_moisturizer"

# ---------------------------
# Feature columns
# ---------------------------
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

# ---------------------------
# Load parquet
# ---------------------------
df = pd.read_parquet(PARQUET_FILE, engine="pyarrow")

# ---------------------------
# Encode categorical features
# ---------------------------
encoders = {}
for col in CATEGORICAL_COLUMNS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ---------------------------
# Convert numeric columns to float & fill NaNs
# ---------------------------
for col in NUMERIC_COLUMNS:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # convert to float, invalid -> NaN
    df[col] = df[col].fillna(df[col].median())          # fill NaNs with median

# ---------------------------
# Prepare features and encode target
# ---------------------------
X = df[FEATURE_COLUMNS]

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df[TARGET].astype(str))

# Save target encoder for later decoding
with open("moisturizer_target_encoder.pkl", "wb") as f:
    pickle.dump(target_encoder, f)

# ---------------------------
# Create LightGBM dataset
# ---------------------------
train_data = lgb.Dataset(X, label=y, free_raw_data=False)

# ---------------------------
# Train model
# ---------------------------
model = lgb.train(
    params={
        "objective": "multiclass",
        "num_class": np.unique(y).size,
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

# ---------------------------
# Save model
# ---------------------------
model.save_model("moisturizer_model.txt")
print("✅ Model saved as moisturizer_model.txt")
