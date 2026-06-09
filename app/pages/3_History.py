import streamlit as st
import sqlite3
import pandas as pd

st.set_page_config(page_title="History", page_icon="📋", layout="wide")
st.title("📋 Analysis History")

def load_history():
    try:
        conn = sqlite3.connect("data/analyses.db")
        df = pd.read_sql("SELECT * FROM analyses ORDER BY created_at DESC LIMIT 50", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

df = load_history()

if df.empty:
    st.info("No analyses yet. Go to Analyze Listing to get started.")
else:
    st.dataframe(df, use_container_width=True)
    st.caption(f"Showing last {len(df)} analyses.")