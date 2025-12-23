import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Path to your cleaned dataset
PARQUET_FILE = "cleaned_skincare_dataset.parquet"

# Columns that are categorical
CATEGORICAL_COLUMNS = [
    "skin_type", "skin_tone", "primary_concern", "secondary_concerns",
    "occupation", "diet_type", "sun_exposure", "sunscreen_usage", "allergies"
]

# Load the dataset
df = pd.read_parquet(PARQUET_FILE, engine="pyarrow")

# Create encoders dictionary
encoders = {}
for col in CATEGORICAL_COLUMNS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Save encoders to a pickle file
with open("feature_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("✅ Feature encoders saved as feature_encoders.pkl")
