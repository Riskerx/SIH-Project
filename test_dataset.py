import pandas as pd

df = pd.read_csv("railway_fittings_data.csv")

print("✅ Dataset loaded successfully!")
print("\nShape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nVendor distribution:\n", df['Vendor'].value_counts())
print("\nFailure rate (%):\n", df['Failure'].value_counts(normalize=True)*100)
