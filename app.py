import streamlit as st
import numpy as np
import joblib
import random

# Load trained Random Forest model
rf = joblib.load("rf_model.pkl")

st.title("Economic Impact Prediction of Disasters")

# --- User inputs ---
year = st.number_input("Year", min_value=1900, max_value=2100, value=2000, step=1)
damage = st.number_input("Damage (in USD)", min_value=0.0, value=0.0, step=1000.0)

# Context-only inputs (not used in prediction)
population_affected = st.number_input("Population Affected", min_value=0, value=0, step=100)
event_duration = st.number_input("Event Duration (days)", min_value=1, value=5, step=1)
event_type = st.selectbox("Event Type", ["Flood", "Storm", "Earthquake", "Wildfire", "Other"])

#-
log_cost = np.log1p(damage)
cost_per_day = damage / max(event_duration, 1) if damage > 0 else 0.0


deaths = 0
input_data = [[deaths, year, cost_per_day, log_cost]]

# --- Prediction ---
if st.button("Predict"):
    prediction = rf.predict(input_data)

    st.success(f"💰 Predicted CPI-Adjusted Cost: **${prediction[0]:,.2f}**")

    
    st.subheader("📊 Details of your input and derived features")
    st.json({
        "Year": year,
        "Event Type": event_type,
        "Damage (USD)": damage,
        "Population Affected": population_affected,
        "Event Duration (days)": event_duration,
        "Injuries (context)": random.randint(0, 500),  # dummy number just for display
        "Cost per Day": round(cost_per_day, 2),
        "Log Cost": round(log_cost, 2),
        "Predicted CPI-Adjusted Cost": round(float(prediction[0]), 2)
    })
