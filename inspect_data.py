import pandas as pd

df = pd.read_csv("data/sign_data.csv")

print("🔍 Total rows in dataset:", len(df))
print("🚫 Rows with missing values (NaNs):", df.isnull().any(axis=1).sum())

# Optional: Show a few rows
print("\n🧾 Sample data:")
print(df.head())
