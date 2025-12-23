import pandas as pd
import os

# Current folder
folder_path = "."

# List all CSV files in folder
all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

dfs = []

for file in all_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Combine all CSVs
data = pd.concat(dfs, ignore_index=True)

# Columns to keep as input features
input_columns = [
    'age', 'skin_type', 'skin_tone', 'primary_concern', 'secondary_concerns',
    'occupation', 'sleep_hours', 'daily_water_intake', 'stress_level',
    'exercise_frequency', 'diet_type', 'junk_food_frequency', 'dairy_intake',
    'sugar_intake', 'sun_exposure', 'sunscreen_usage', 'allergies'
]

# Columns to keep as output labels
label_columns = [
    "rec_moisturizer",
    "rec_serum",
    "rec_sunscreen",
    "rec_treatment",
    "rec_ingredients",
    "avoid_ingredients",
    "rec_routine_type"
]

# Keep only inputs + labels
columns_to_keep = input_columns + label_columns
data = data[columns_to_keep]

# Remove rows that are completely empty in labels
data = data.dropna(subset=label_columns, how="all")

# Optional: fill missing input values with placeholder or mode
for col in input_columns:
    if data[col].dtype == object:
        data[col] = data[col].fillna("Unknown")
    else:
        data[col] = data[col].fillna(0)

# Check the first few rows
print(data.head())

# Save cleaned dataset for training
data.to_csv("cleaned_skincare_dataset.csv", index=False)
print("Cleaned dataset saved as cleaned_skincare_dataset.csv")
