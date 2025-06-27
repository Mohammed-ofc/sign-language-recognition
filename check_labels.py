import pandas as pd

df = pd.read_csv("data/sign_data.csv", header=None)
print("\n🔠 Sample count per gesture label:")
print(df[df.columns[-1]].value_counts())

