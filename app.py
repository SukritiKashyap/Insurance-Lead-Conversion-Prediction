import streamlit as st
import pandas as pd
import joblib

# ================================
# LOAD MODEL
# ================================
model = joblib.load("models/improved_model.pkl")

st.set_page_config(page_title="Insurance Conversion Predictor", layout="wide")

st.title("💼 Insurance Lead Conversion & Revenue Predictor")

st.markdown("Predict conversion probability and estimate expected revenue for a lead.")

# ================================
# USER INPUTS
# ================================
st.sidebar.header("Input Lead Details")

age = st.sidebar.slider("Age", 18, 80, 30)
credit_score = st.sidebar.slider("Credit Score", 300, 900, 650)
website_visits = st.sidebar.slider("Website Visits", 0, 50, 5)
inquiries = st.sidebar.slider("Inquiries", 0, 20, 2)
quotes = st.sidebar.slider("Quotes Requested", 0, 10, 1)
premium = st.sidebar.slider("Premium Amount", 1000, 100000, 20000)
time_since_contact = st.sidebar.slider("Days Since First Contact", 0, 100, 10)

# ================================
# FEATURE ENGINEERING (MATCH MODEL)
# ================================
engagement_score = website_visits + inquiries + quotes
conversion_intent = (quotes * 2) + inquiries

# ================================
# CREATE INPUT DATAFRAME
# ================================
input_data = pd.DataFrame({
    "Age": [age],
    "Credit_Score": [credit_score],
    "Website_Visits": [website_visits],
    "Inquiries": [inquiries],
    "Quotes_Requested": [quotes],
    "Premium_Amount": [premium],
    "Time_Since_First_Contact": [time_since_contact],
    "engagement_score": [engagement_score],
    "conversion_intent": [conversion_intent]
})

# ================================
# PREDICTION
# ================================
prob = model.predict_proba(input_data)[0][1]
prediction = model.predict(input_data)[0]

expected_revenue = prob * premium

# ================================
# DISPLAY RESULTS
# ================================
st.subheader("📊 Prediction Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Conversion Prediction", "Yes" if prediction == 1 else "No")

with col2:
    st.metric("Conversion Probability", f"{prob:.2f}")

with col3:
    st.metric("Expected Revenue", f"₹ {expected_revenue:,.0f}")

# ================================
# INTERPRETATION
# ================================
st.subheader("📌 Interpretation")

if prob > 0.7:
    st.success("🔥 High-value lead — prioritize immediately.")
elif prob > 0.4:
    st.warning("⚠️ Medium potential — follow up recommended.")
else:
    st.error("❌ Low conversion likelihood — low priority.")

# ================================
# BUSINESS INSIGHT PANEL
# ================================
st.subheader("📈 Business Insight")

st.write(f"""
- Engagement Score: **{engagement_score}**
- Conversion Intent Score: **{conversion_intent}**

Leads with higher engagement and quote activity typically show stronger conversion behavior.
""")
