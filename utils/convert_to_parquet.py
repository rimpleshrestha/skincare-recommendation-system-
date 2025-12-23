import pandas as pd

df = pd.read_csv(
    "cleaned_skincare_dataset.csv",
    low_memory=False
)

df.to_parquet(
    "cleaned_skincare_dataset.parquet",
    engine="pyarrow",
    compression="snappy"
)

print("DONE: CSV converted to Parquet")
