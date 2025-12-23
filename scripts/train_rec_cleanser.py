import pandas as pd

PARQUET_FILE = "cleaned_skincare_dataset.parquet"

df = pd.read_parquet(PARQUET_FILE, engine="pyarrow")

def assign_cleanser(row):
    skin_type = str(row["skin_type"]).lower()
    concern = str(row["primary_concern"]).lower()
    allergies = str(row["allergies"]).lower()

    if "sensitive" in skin_type or "fragrance" in allergies:
        return "Gentle pH-Balanced Cleanser"

    if "oily" in skin_type or "acne" in concern:
        return "Salicylic Acid Gel Cleanser"

    if "dry" in skin_type:
        return "Cream Cleanser"

    if "combination" in skin_type:
        return "Foaming Cleanser"

    return "Mild Daily Cleanser"

df["rec_cleanser"] = df.apply(assign_cleanser, axis=1)

# Save back to parquet
df.to_parquet(PARQUET_FILE, engine="pyarrow")

print("✅ rec_cleanser column created successfully")
print(df["rec_cleanser"].value_counts())
