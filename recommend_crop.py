import joblib

# Load trained model
model = joblib.load('crop_recommendation_model.pkl')

# Get user input
print("Enter the following values:")
N = float(input("Nitrogen (N): "))
P = float(input("Phosphorus (P): "))
K = float(input("Potassium (K): "))
temp = float(input("Temperature (°C): "))
humidity = float(input("Humidity (%): "))
ph = float(input("pH: "))
rainfall = float(input("Rainfall (mm): "))

# Predict
features = [[N, P, K, temp, humidity, ph, rainfall]]
prediction = model.predict(features)

print(f"\n✅ Recommended Crop: {prediction[0]}")
