import streamlit as st
import sqlite3
import json
import os
from datetime import datetime
from utils.api_client import predict_price, analyze_full_listing, upload_image

st.set_page_config(page_title="Trust Report", page_icon="🛡️", layout="wide")
st.title("🛡️ Vehicle Trust Report")
st.caption("Combines price prediction, NLP fraud detection, and image quality into a single trust score.")

# ------------------------------------------------------------------ #
# DB setup for full trust reports
# ------------------------------------------------------------------ #
def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/analyses.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trust_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            make TEXT, model TEXT, year INTEGER,
            listed_price REAL, predicted_price REAL,
            trust_score REAL, tier TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_report(make, model, year, listed_price, predicted_price, trust_score, tier):
    conn = sqlite3.connect("data/analyses.db")
    conn.execute(
        """INSERT INTO trust_reports 
        (make, model, year, listed_price, predicted_price, trust_score, tier, created_at) 
        VALUES (?,?,?,?,?,?,?,?)""",
        (make, model, year, listed_price, predicted_price, trust_score, tier, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

init_db()

# ------------------------------------------------------------------ #
# Input form
# ------------------------------------------------------------------ #
st.subheader("1️⃣ Vehicle Details")
col1, col2 = st.columns(2)
with col1:
    make = st.text_input("Make", value="Ford")
    year = st.number_input("Year", min_value=1995, max_value=2024, value=2018)
    km_driven = st.number_input("KM Driven", min_value=0, max_value=500000, value=45000, step=1000)
    listed_price = st.number_input("Listed Price ($)", min_value=0, value=18000, step=500)

with col2:
    model_name = st.text_input("Model", value="f-150")
    transmission = st.selectbox("Transmission", ["automatic", "manual"])
    condition = st.slider("Condition Score (1-49)", 1, 49, 35)
    sale_month = st.selectbox("Month", list(range(1, 13)), index=5)

st.subheader("2️⃣ Seller Description")
description = st.text_area(
    "Paste the listing description",
    value="Well maintained vehicle, single owner, full service history. Test drive welcome.",
    height=100
)

st.subheader("3️⃣ Vehicle Photo (Optional)")
uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, width=300, caption="Preview")

analyze_btn = st.button("🛡️ Generate Trust Report", use_container_width=True, type="primary")

# ------------------------------------------------------------------ #
# Run analysis
# ------------------------------------------------------------------ #
if analyze_btn:
    image_filename = None

    if uploaded_file:
        with st.spinner("Analyzing image..."):
            img_result = upload_image(uploaded_file)
            if "error" not in img_result:
                image_filename = uploaded_file.name

    with st.spinner("Running full trust analysis..."):
        result = analyze_full_listing(
            make, model_name, year, km_driven, listed_price,
            transmission, float(condition), sale_month, description,
            image_filename
        )

    if "error" in result:
        st.error(f"Error: {result['error']}")
    else:
        trust = result["trust_report"]
        score = trust["trust_score"]
        tier = trust["tier"]

        st.markdown("---")
        st.subheader("📋 Trust Report")

        # Tier color coding
        tier_colors = {
            "TRUSTWORTHY": "🟢",
            "CAUTION": "🟡",
            "HIGH RISK": "🟠",
            "LIKELY SCAM": "🔴"
        }

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Trust Score", f"{score}/100")
        with col2:
            st.metric("Risk Tier", f"{tier_colors.get(tier, '')} {tier}")
        with col3:
            st.metric("Predicted Fair Price", f"${result['predicted_price']:,.0f}")

        st.progress(int(score) / 100)

        # Component breakdown
        st.markdown("---")
        st.subheader("📊 Risk Breakdown")
        risks = trust["component_risks"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Price Anomaly Risk", f"{risks['price_anomaly_risk']}/100")
        col2.metric("NLP Fraud Risk", f"{risks['nlp_fraud_risk']}/100")
        col3.metric("Image Quality Risk", f"{risks['image_quality_risk']}/100")

        # Explanations
        st.markdown("---")
        st.subheader("💡 Explanation")
        for exp in trust["explanations"]:
            severity = exp["severity"]
            icon = {"HIGH": "🔴", "MEDIUM": "🟠", "NONE": "🟢"}.get(severity, "⚪")
            with st.container(border=True):
                st.markdown(f"{icon} **{exp['flag']}** — {exp['message']}")
                if exp.get("suggestion"):
                    st.caption(f"💬 {exp['suggestion']}")

        # Save
        save_report(make, model_name, year, listed_price,
                   result["predicted_price"], score, tier)
        st.success("✓ Trust report saved to history.")

