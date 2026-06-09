import streamlit as st
import sqlite3
import json
import os
from datetime import datetime
from app.utils.api_client import predict_price

st.set_page_config(page_title="Analyze Listing", page_icon="🔍", layout="wide")
st.title("🔍 Analyze a Vehicle Listing")

# ------------------------------------------------------------------ #
# SQLite setup — store every analysis
# ------------------------------------------------------------------ #
def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/analyses.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            make TEXT,
            model TEXT,
            year INTEGER,
            km_driven INTEGER,
            predicted_price REAL,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_analysis(make, model, year, km, price):
    conn = sqlite3.connect("data/analyses.db")
    conn.execute(
        "INSERT INTO analyses (make, model, year, km_driven, predicted_price, created_at) VALUES (?,?,?,?,?,?)",
        (make, model, year, km, price, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

init_db()

# ------------------------------------------------------------------ #
# Input form
# ------------------------------------------------------------------ #
with st.form("listing_form"):
    st.subheader("Vehicle Details")
    col1, col2 = st.columns(2)

    with col1:
        make = st.text_input("Make", value="Ford")
        year = st.number_input("Year", min_value=1995, max_value=2024, value=2018)
        km_driven = st.number_input("KM Driven", min_value=0,
                                     max_value=500000, value=45000, step=1000)

    with col2:
        model_name = st.text_input("Model", value="f-150")
        transmission = st.selectbox("Transmission", ["automatic", "manual"])
        condition = st.slider("Condition Score (1=Poor, 49=Excellent)", 1, 49, 35)

    sale_month = st.selectbox("Sale Month", list(range(1, 13)), index=5)
    submitted = st.form_submit_button("🔍 Analyze Listing", use_container_width=True)

# ------------------------------------------------------------------ #
# Result display
# ------------------------------------------------------------------ #
if submitted:
    with st.spinner("Analyzing..."):
        result = predict_price(make, model_name, year, km_driven,
                               transmission, float(condition), sale_month)

    if "error" in result:
        st.error(f"Error: {result['error']}")
    else:
        price = result["predicted_price_usd"]
        st.markdown("---")
        st.subheader("📊 Analysis Result")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Fair Price", f"${price:,.0f}")
        with col2:
            st.metric("Vehicle Age", f"{2024 - year} years")
        with col3:
            condition_label = "Low" if condition <= 20 else ("Mid" if condition <= 35 else "High")
            st.metric("Condition Band", condition_label)

        # Price context
        st.markdown("---")
        st.subheader("💡 What This Means")

        if condition <= 20:
            st.warning("⚠️ Low condition score. Expect higher maintenance costs.")
        elif condition >= 40:
            st.success("✅ High condition score. Vehicle appears well maintained.")

        if km_driven > 150000:
            st.warning(f"⚠️ High mileage ({km_driven:,} km). Factor in wear-related costs.")

        age = 2024 - year
        if age > 10:
            st.info(f"ℹ️ Vehicle is {age} years old. Parts availability may be limited.")

        # Save to history
        save_analysis(make, model_name, year, km_driven, price)
        st.caption("✓ Analysis saved to history.")