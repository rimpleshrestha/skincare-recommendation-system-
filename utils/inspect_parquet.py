import pandas as pd

df = pd.read_parquet(
    "cleaned_skincare_dataset.parquet",
    engine="pyarrow"
)

print("Total rows:", len(df))
print("Columns:")
print(df.columns.tolist())

print("\nSample rows:")
print(df.head(3))
