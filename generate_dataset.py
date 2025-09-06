import pandas as pd
import random
from faker import Faker
import numpy as np

fake = Faker("en_IN")   # Indian context
random.seed(42)
np.random.seed(42)

# Vendors (mock realistic codes/names)
vendors = [
    ("V001", "Tata Steel"),
    ("V002", "Jindal Rail Infra"),
    ("V003", "SAIL"),
    ("V004", "Bharat Forge"),
    ("V005", "Reliance Infra"),
]

# Track sections
track_sections = ["Heavy Load Zone", "Coastal Area", "Hilly Terrain", "Urban Metro", "Plain"]

data = []

for i in range(1000):
    qr_id = f"QR{i+1:04d}"

    # Pick random vendor
    vendor_code, vendor_name = random.choice(vendors)

    lot_number = f"LOT{random.randint(1000,9999)}"
    manufacture_date = fake.date_between(start_date="-2y", end_date="today")
    warranty_years = random.choice([1, 2, 3, 5])
    warranty_expiry = manufacture_date.replace(year=manufacture_date.year + warranty_years)

    # Inspections: 2–5 per fitting
    num_inspections = random.randint(2, 5)
    inspections = []
    for j in range(num_inspections):
        insp_date = fake.date_between(start_date=manufacture_date, end_date="today")
        result = random.choices(["Pass", "Fail"], weights=[0.9, 0.1])[0]
        wear = round(np.clip(np.random.normal(loc=15, scale=10), 0, 100), 2)  # wear% around 15
        inspections.append({"date": insp_date, "result": result, "wear": wear})

    # Failure report (rare)
    failure = random.choices(["Yes", "No"], weights=[0.05, 0.95])[0]
    failure_time = fake.date_between(start_date=manufacture_date, end_date="today") if failure == "Yes" else None

    # Track info
    track_info = random.choice(track_sections)

    data.append({
        "QR_ID": qr_id,
        "Vendor_Code": vendor_code,
        "Vendor_Name": vendor_name,
        "Lot_Number": lot_number,
        "Date_of_Manufacture": manufacture_date,
        "Warranty_Expiry": warranty_expiry,
        "Num_Inspections": num_inspections,
        "Inspection_Records": inspections,
        "In_Service_Failure": failure,
        "Failure_Date": failure_time,
        "Track_Section": track_info,
    })

# Save in flattened form
df = pd.DataFrame(data)

# Convert inspection records to JSON-like string for easy storage
df["Inspection_Records"] = df["Inspection_Records"].apply(lambda x: str(x))

df.to_csv("railway_fittings_data.csv", index=False)
df.to_excel("railway_fittings_data.xlsx", index=False)

print("✅ Generated 1000 realistic railway fitting records!")
