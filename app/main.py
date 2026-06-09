import streamlit as st
from app.utils.api_client import health_check

st.set_page_config(
    page_title="Vehicle Trust Platform",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Vehicle Trust Intelligence Platform")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model Version", "XGBoost v1")

with col2:
    st.metric("Training R²", "0.906")

with col3:
    status = health_check()
    api_status = "🟢 Online" if status.get("status") == "ok" else "🔴 Offline"
    st.metric("API Status", api_status)

st.markdown("---")

st.markdown("""
### What this platform does

- **Price Prediction** — Estimates fair market value for used vehicle listings
- **Market Analytics** — Explore price trends across makes, models, and conditions
- **History** — Review all past analyses
- **Fraud Detection** — Analyze suspicious seller descriptions

### How to use

1. Go to **Analyze Listing** in the sidebar
2. Enter vehicle details
3. Get an instant price estimate
4. Use fraud detection to analyze listing text

> Navigate using the sidebar on the left.
""")