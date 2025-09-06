import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("railway_fittings_data.csv")

print("✅ Dataset loaded:", df.shape, "rows")

# --- Feature Engineering ---
# Convert categorical values into numbers
df_encoded = df.copy()
df_encoded['Failure_Binary'] = df_encoded['Failure'].map({"Yes": 1, "No": 0})

# Extract number of inspections and average wear %
def parse_inspections(record):
    parts = record.split(";")
    num_inspections = len(parts)
    wear_values = []
    for p in parts:
        try:
            wear = int(p.split(":")[-1].replace("%",""))
            wear_values.append(wear)
        except:
            pass
    avg_wear = sum(wear_values)/len(wear_values) if wear_values else 0
    return num_inspections, avg_wear

df_encoded[['Num_Inspections','Avg_Wear']] = df_encoded['Inspection_Records'].apply(
    lambda x: pd.Series(parse_inspections(x))
)

# Features for anomaly detection
features = df_encoded[['Num_Inspections','Avg_Wear','Failure_Binary']]

# --- Isolation Forest Model ---
model = IsolationForest(contamination=0.1, random_state=42)
df_encoded['Anomaly'] = model.fit_predict(features)

# Map anomaly result (-1 = anomaly, 1 = normal)
df_encoded['Anomaly'] = df_encoded['Anomaly'].map({-1: "Anomaly", 1: "Normal"})

# --- Results ---
print("\n🔍 Anomaly Detection Results:")
print(df_encoded[['QR_ID','Vendor','Lot_Number','Num_Inspections','Avg_Wear','Failure','Anomaly']].head(15))

# Save results
df_encoded.to_csv("fittings_with_anomalies.csv", index=False)
print("\n✅ Results saved to fittings_with_anomalies.csv")

# --- Plot (Optional) ---
plt.figure(figsize=(6,4))
colors = df_encoded['Anomaly'].map({"Normal":"blue","Anomaly":"red"})
plt.scatter(df_encoded['Avg_Wear'], df_encoded['Num_Inspections'], c=colors, alpha=0.6)
plt.xlabel("Average Wear (%)")
plt.ylabel("Number of Inspections")
plt.title("Anomaly Detection on Railway Fittings")
plt.show()
