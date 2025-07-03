import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model
model = joblib.load("crop_recommendation_model.pkl")

# Page config
st.set_page_config(page_title="Crop Recommendation", layout="centered")
st.title("🌱 Crop Recommendation System")
st.write("Enter soil and environmental conditions below to get a suitable crop suggestion.")

# Input form
with st.form("input_form"):
    N = st.number_input("Nitrogen (N)", min_value=0.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0)
    K = st.number_input("Potassium (K)", min_value=0.0)
    temperature = st.number_input("Temperature (°C)", min_value=0.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0)
    ph = st.number_input("pH", min_value=0.0, max_value=14.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

    submitted = st.form_submit_button("🌾 Recommend Crop")

# After submit
if submitted:
    # Predict
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    st.success(f"✅ Recommended Crop: **{prediction[0].capitalize()}**")

    # Bar chart of input values
    features = {
        "Nitrogen (N)": N,
        "Phosphorus (P)": P,
        "Potassium (K)": K,
        "Temperature (°C)": temperature,
        "Humidity (%)": humidity,
        "pH": ph,
        "Rainfall (mm)": rainfall
    }

    df = pd.DataFrame(list(features.items()), columns=["Feature", "Value"])
    st.subheader("🌿 Input Summary")
    st.bar_chart(df.set_index("Feature"))

    # Optional tip
    tips = [
        "Use organic compost for better soil structure.",
        "Avoid excessive irrigation to prevent nutrient loss.",
        "Check pH every season for ideal crop matching.",
        "Crop rotation improves soil fertility!"
    ]
    st.info(np.random.choice(tips))
