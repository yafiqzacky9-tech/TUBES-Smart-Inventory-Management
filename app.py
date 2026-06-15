import streamlit as st
import joblib
import numpy as np

# ======================
# PAGE CONFIG
# ======================

st.set_page_config(
    page_title="Smart Inventory Management",
    page_icon="📦",
    layout="centered"
)

# ======================
# LOAD MODEL
# ======================

model = joblib.load("model_inventory.pkl")
scaler = joblib.load("scaler.pkl")

# ======================
# HEADER
# ======================

st.title("📦 Smart Inventory Management")

st.markdown("""
### 🤖 Inventory Reorder Prediction System

Predict whether an item needs to be reordered based on stock and sales data.
""")

st.divider()

# ======================
# INPUT FORM
# ======================

col1, col2 = st.columns(2)

with col1:
    current_stock = st.number_input(
        "📦 Current Stock",
        min_value=0
    )

    minimum_stock = st.number_input(
        "📉 Minimum Stock",
        min_value=0
    )

with col2:
    avg_daily_sales = st.number_input(
        "📊 Average Daily Sales",
        min_value=0
    )

    lead_time_days = st.number_input(
        "🚚 Lead Time (Days)",
        min_value=0
    )

st.divider()

# ======================
# PREDICTION
# ======================

if st.button("🔍 Predict Reorder Status"):

    data = np.array([[
        current_stock,
        minimum_stock,
        avg_daily_sales,
        lead_time_days
    ]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    st.divider()

    if prediction[0] == 1:

        st.error(
            "⚠️ NEED REORDER\n\nCurrent inventory may not be sufficient for future demand."
        )

    else:

        st.success(
            "✅ STOCK IS SUFFICIENT\n\nNo reorder is required at this time."
        )

# ======================
# FOOTER
# ======================

st.divider()

st.caption(
    "📚 AI & Big Data Final Project | Inventory Reorder Prediction"
)
