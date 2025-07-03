import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("crop_recommendation_model.pkl")

# Page setup
st.set_page_config(page_title="Crop Recommendation", layout="centered")
st.title("🌱 Crop Recommendation System")
st.write("Enter environmental details to get a suitable crop suggestion.")

# Form input
with st.form("input_form"):
    N = st.number_input("Nitrogen (N)", min_value=0.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0)
    K = st.number_input("Potassium (K)", min_value=0.0)
    temperature = st.number_input("Temperature (°C)", min_value=0.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0)
    ph = st.number_input("pH", min_value=0.0, max_value=14.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

    submitted = st.form_submit_button("🌾 Recommend Crop")

# Prediction
if submitted:
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    st.success(f"✅ Recommended Crop: **{prediction[0].capitalize()}**")
