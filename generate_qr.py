import qrcode
import pandas as pd
import os

# Load dataset
df = pd.read_csv("fittings_with_anomalies.csv")

# Create folder
os.makedirs("qr_codes", exist_ok=True)

# Generate QR for each fitting
for idx, row in df.iterrows():
    qr_id = row["QR_ID"]
    img = qrcode.make(qr_id)
    img.save(f"qr_codes/{qr_id}.png")

print("✅ QR codes generated in qr_codes/ folder")
