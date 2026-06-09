import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Market Analytics", page_icon="📊", layout="wide")
st.title("📊 Market Analytics")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/vehicle_sales_features.csv")

df = load_data()

st.markdown("---")

# ------------------------------------------------------------------ #
# Chart 1 — Price distribution
# ------------------------------------------------------------------ #
st.subheader("Price Distribution")
fig = px.histogram(df, x="price", nbins=80,
                   title="Distribution of Vehicle Prices",
                   color_discrete_sequence=["steelblue"])
fig.update_layout(xaxis_title="Price (USD)", yaxis_title="Count")
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ #
# Chart 2 — Top makes by volume
# ------------------------------------------------------------------ #
st.subheader("Top 15 Makes by Listing Volume")
top_makes = df["make"].value_counts().head(15).reset_index()
top_makes.columns = ["make", "count"]
fig2 = px.bar(top_makes, x="count", y="make", orientation="h",
              color_discrete_sequence=["coral"])
fig2.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------------ #
# Chart 3 — Price vs Age filtered by make
# ------------------------------------------------------------------ #
st.subheader("Price vs Vehicle Age by Make")
top_10_makes = df["make"].value_counts().head(10).index.tolist()
selected_make = st.selectbox("Select Make", top_10_makes)

filtered = df[df["make"] == selected_make].sample(n=min(2000, len(df)))
fig3 = px.scatter(filtered, x="vehicle_age", y="price",
                  color="transmission",
                  title=f"{selected_make} — Price vs Age",
                  opacity=0.4)
st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------------------------ #
# Chart 4 — Median price by condition band
# ------------------------------------------------------------------ #
st.subheader("Median Price by Condition Band")
cond_price = df.groupby("condition_band")["price"].median().reset_index()
fig4 = px.bar(cond_price, x="condition_band", y="price",
              color_discrete_sequence=["mediumseagreen"])
st.plotly_chart(fig4, use_container_width=True)