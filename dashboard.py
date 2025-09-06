# dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import ast
from datetime import datetime

# Optional import for webcam; we will handle the case it's not installed.
try:
    from streamlit_webrtc import webrtc_streamer
    STREAMLIT_WEBRTC_AVAILABLE = True
except Exception:
    STREAMLIT_WEBRTC_AVAILABLE = False

# --- Page Config ---
st.set_page_config(
    page_title="Railway Fittings Dashboard",
    layout="wide",
    page_icon="🚆"
)

# --- Helper functions ---
def safe_parse_inspections(rec):
    if pd.isna(rec):
        return 0, 0.0

    if isinstance(rec, (list, tuple)):
        rec_list = rec
    else:
        rec_str = str(rec)
        try:
            rec_list = ast.literal_eval(rec_str)
        except Exception:
            parts = [p.strip() for p in rec_str.split(";") if p.strip()]
            rec_list = []
            for p in parts:
                try:
                    parts2 = p.split(":")
                    if len(parts2) >= 3:
                        wear = parts2[-1].strip().replace("%", "")
                        wear = float(wear)
                        rec_list.append({"wear": wear})
                except Exception:
                    pass

    wear_vals = []
    for item in rec_list:
        if isinstance(item, dict) and "wear" in item:
            try:
                wear_vals.append(float(item["wear"]))
            except Exception:
                pass
        else:
            try:
                num = float(''.join(ch for ch in str(item) if (ch.isdigit() or ch == '.')))
                wear_vals.append(num)
            except Exception:
                pass

    num_inspections = len(rec_list)
    avg_wear = float(np.mean(wear_vals)) if wear_vals else 0.0
    return num_inspections, round(avg_wear, 2)

@st.cache_data
def load_and_prepare(path="fittings_with_anomalies.csv"):
    df = pd.read_csv(path)

    if "Date_of_Manufacture" in df.columns:
        try:
            df["Date_of_Manufacture"] = pd.to_datetime(df["Date_of_Manufacture"])
        except Exception:
            df["Date_of_Manufacture"] = pd.to_datetime(df["Date_of_Manufacture"], errors="coerce")

    if "Avg_Wear" not in df.columns or "Num_Inspections" not in df.columns:
        num_ins, avg_wears = [], []
        for rec in df.get("Inspection_Records", pd.Series([None]*len(df))):
            n, a = safe_parse_inspections(rec)
            num_ins.append(n)
            avg_wears.append(a)
        df["Num_Inspections"] = df.get("Num_Inspections", pd.Series(num_ins))
        df["Avg_Wear"] = df.get("Avg_Wear", pd.Series(avg_wears))

    if "Failure" in df.columns:
        df["Failure"] = df["Failure"].astype(str).str.strip().replace({"True": "Yes", "False":"No"})
    else:
        df["Failure"] = "No"

    if "Anomaly" not in df.columns:
        df["Anomaly"] = "Normal"

    if "Vendor" not in df.columns and "Vendor_Name" in df.columns:
        df["Vendor"] = df["Vendor_Name"]

    return df

df = load_and_prepare()

# --- Header ---
st.title("🚆 Railway Fittings Quality Monitoring")
st.markdown("A smart AI-enabled system for **real-time anomaly detection, vendor insights, and QR-based tracking**.")

# --- Tabs for navigation ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🏭 Vendor Insights",
    "⚠️ Anomaly Detection",
    "📷 QR Scanner",
    "🤖 Vendor Decision Support"
])

# ================== TAB 1: Overview ==================
with tab1:
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", df.shape[0])
    col2.metric("Vendors", int(df['Vendor'].nunique()))
    col3.metric("Failures", int((df['Failure'] == "Yes").sum()))

    st.dataframe(df.head(15), width="stretch")

    st.subheader("Failure Distribution")
    failure_counts = df['Failure'].value_counts(normalize=True) * 100
    st.bar_chart(failure_counts)

# ================== TAB 2: Vendor Insights ==================
with tab2:
    st.subheader("Vendor Performance Analysis")
    selected_vendor = st.selectbox("Select Vendor", df['Vendor'].unique())

    vendor_data = df[df['Vendor'] == selected_vendor].copy()

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"### Last 10 Records for {selected_vendor}")
        st.dataframe(vendor_data.sort_values("Date_of_Manufacture", ascending=False).head(10), width="stretch")

    with col2:
        if "Date_of_Manufacture" in vendor_data.columns:
            failures_over_time = vendor_data.groupby(vendor_data["Date_of_Manufacture"].dt.date)["Failure"].apply(
                lambda x: (x == "Yes").sum()
            ).reset_index(name="Failures")
            failures_over_time = failures_over_time.set_index("Date_of_Manufacture")
            if not failures_over_time.empty:
                st.line_chart(failures_over_time)

    if "Avg_Wear" in vendor_data.columns:
        st.subheader("Wear Trend Over Time")
        wear_trend = vendor_data.groupby(vendor_data["Date_of_Manufacture"].dt.date)["Avg_Wear"].mean().reset_index(name="Avg_Wear")
        wear_trend = wear_trend.set_index("Date_of_Manufacture")
        if not wear_trend.empty:
            st.line_chart(wear_trend)

# ================== TAB 3: Anomaly Detection ==================
with tab3:
    st.subheader("AI-Powered Anomaly Detection")

    option = st.radio("Filter by status:", ["All", "Normal", "Anomaly"], horizontal=True)
    filtered = df if option == "All" else df[df['Anomaly'] == option]
    st.dataframe(filtered[['QR_ID','Vendor','Lot_Number','Num_Inspections','Avg_Wear','Failure','Anomaly']], width="stretch")

    st.subheader("Anomalies Visualization")
    fig, ax = plt.subplots()
    colors = filtered['Anomaly'].map({"Normal":"blue","Anomaly":"red"})
    ax.scatter(filtered['Avg_Wear'], filtered['Num_Inspections'], c=colors, alpha=0.6)
    ax.set_xlabel("Average Wear (%)")
    ax.set_ylabel("Number of Inspections")
    ax.set_title("Anomaly Detection on Railway Fittings")
    st.pyplot(fig)

# ================== TAB 4: QR Code Scanner ==================
with tab4:
    st.subheader("QR Code Based Lookup & Vendor Analysis")

    scan_mode = st.radio("Choose Scan Mode:", ["Upload Image", "Live Camera (if available)"])

    qr_detector = cv2.QRCodeDetector()

    def fetch_and_show(qr_data):
        st.success(f"✅ QR Code Data: {qr_data}")
        record = df[df["QR_ID"] == qr_data]
        if not record.empty:
            st.write("### Fitting Details")
            st.dataframe(record, width="stretch")

            if record["Anomaly"].iloc[0] == "Anomaly":
                st.error("⚠️ This fitting is flagged as ANOMALY. Inspect / Replace as necessary.")
            else:
                st.success("✅ This fitting appears normal.")

            qr_vendor = record["Vendor"].iloc[0]
            vendor_data = df[df["Vendor"] == qr_vendor]
            total_supplied = vendor_data.shape[0]
            total_failures = int((vendor_data["Failure"] == "Yes").sum())
            failure_rate = (total_failures / total_supplied) * 100 if total_supplied > 0 else 0.0

            st.markdown(f"""
            ### 🏭 Vendor Analysis: **{qr_vendor}**
            - 📦 Total Supplied: **{total_supplied} items**
            - ❌ Failures: **{total_failures} items** ({failure_rate:.2f}% defective)
            """)

            if failure_rate < 5:
                st.success("✅ Vendor Quality is Good — Safe to Buy")
            elif failure_rate < 15:
                st.warning("⚠️ Vendor has Moderate Failures — Caution Needed")
            else:
                st.error("❌ High Failure Rate — Avoid Purchasing")

            st.write("### Vendor Failure vs Working Items")
            failure_summary = vendor_data["Failure"].value_counts()
            st.bar_chart(failure_summary)

            st.write("### Recent Vendor History")
            vendor_history = vendor_data.sort_values("Date_of_Manufacture", ascending=False).head(10)
            st.dataframe(vendor_history, width="stretch")
        else:
            st.error("❌ No record found for this QR ID in dataset")

    if scan_mode == "Upload Image":
        uploaded_file = st.file_uploader("Upload a QR code image", type=["png","jpg","jpeg"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded QR", width="stretch")
            img = Image.open(uploaded_file).convert("RGB")
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            data, points, _ = qr_detector.detectAndDecode(img_cv)
            if data:
                fetch_and_show(data)
            else:
                st.error("❌ Could not decode QR code. Try a clearer image or different angle.")
    else:
        if not STREAMLIT_WEBRTC_AVAILABLE:
            st.warning("Live camera scanning requires the 'streamlit-webrtc' package. Install it with:\n\npip install streamlit-webrtc av")
            st.info("You can still use 'Upload Image' mode.")
        else:
            st.markdown("🔎 Point your camera at a QR code to scan in real-time. (Allow camera access in browser)")

            def video_frame_callback(frame):
                img = frame.to_ndarray(format="bgr24")
                data, points, _ = qr_detector.detectAndDecode(img)
                if data:
                    if points is not None:
                        pts = points.astype(int).reshape((-1,2))
                        for j in range(len(pts)):
                            cv2.line(img, tuple(pts[j]), tuple(pts[(j+1) % len(pts)]), (0,255,0), 2)
                    cv2.putText(img, data, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    st.session_state['last_qr'] = data
                return img

            webrtc_cfg = webrtc_streamer(
                key="qr-live",
                video_frame_callback=video_frame_callback,
                media_stream_constraints={"video": True, "audio": False}
            )

            last = st.session_state.get('last_qr', None)
            if last:
                fetch_and_show(last)

# ================== TAB 5: Vendor Decision Support ==================
with tab5:
    st.subheader("AI Recommendation: Should We Buy From This Vendor?")

    vendor_stats = df.groupby("Vendor").agg(
        total=("QR_ID", "count"),
        failures=("Failure", lambda x: (x == "Yes").sum()),
        anomalies=("Anomaly", lambda x: (x == "Anomaly").sum()),
        avg_wear=("Avg_Wear", "mean")
    ).reset_index()

    vendor_stats["FailureRate"] = vendor_stats["failures"] / vendor_stats["total"]
    vendor_stats["AnomalyRate"] = vendor_stats["anomalies"] / vendor_stats["total"]

    vendor_stats["RiskScore"] = (
        0.5 * vendor_stats["FailureRate"] +
        0.3 * vendor_stats["AnomalyRate"] +
        0.2 * (vendor_stats["avg_wear"] / 100)
    )

    def get_recommendation(score):
        if score < 0.2:
            return "✅ Safe to Buy"
        elif score < 0.4:
            return "⚠️ Caution"
        else:
            return "❌ Avoid"

    vendor_stats["Recommendation"] = vendor_stats["RiskScore"].apply(get_recommendation)

    st.write("### Vendor Risk Ranking")
    st.dataframe(vendor_stats.sort_values("RiskScore"), width="stretch")

    selected_vendor = st.selectbox("Select Vendor for Detailed Report", vendor_stats["Vendor"].unique())
    vendor_row = vendor_stats[vendor_stats["Vendor"] == selected_vendor].iloc[0]

    st.markdown(f"""
    ### 🏭 Vendor: **{selected_vendor}**
    - 📦 Total Supplies: **{vendor_row['total']}**
    - ❌ Failures: **{vendor_row['failures']}** ({vendor_row['FailureRate']*100:.1f}%)
    - 🚨 Anomalies: **{vendor_row['anomalies']}** ({vendor_row['AnomalyRate']*100:.1f}%)
    - 🔧 Avg Wear: **{vendor_row['avg_wear']:.2f}%**
    - 📊 Risk Score: **{vendor_row['RiskScore']:.2f}**
    - 🧠 Recommendation: **{vendor_row['Recommendation']}**
    """)
