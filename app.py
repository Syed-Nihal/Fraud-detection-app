import streamlit as st
import numpy as np
import joblib

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# UI HEADER
# -----------------------------
st.title("💳 Credit Card Fraud Detection")

st.info("This system predicts whether a transaction is fraudulent using Machine Learning.")

st.markdown("### Enter transaction details:")

# -----------------------------
# INPUTS
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("💰 Transaction Amount", min_value=0.0, value=100.0)

with col2:
    time = st.number_input("⏱️ Time", min_value=0.0, value=0.0)

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("🔍 Check Fraud"):
    
    # Create input (same format as training)
    features = [0]*28
    input_data = [time] + features + [amount]

    data = np.array(input_data).reshape(1, -1)
    data = scaler.transform(data)

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    # -----------------------------
    # RESULT DISPLAY
    # -----------------------------
    st.markdown("## Result:")

    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction Detected!\n\nConfidence: {probability:.2f}")
    else:
        st.success(f"✅ Normal Transaction\n\nConfidence: {1 - probability:.2f}")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Built using Machine Learning (Random Forest / XGBoost + SMOTE)")